#=
Copyright 2019, Chris Coey and contributors

advanced scaling point based stepping routine
=#

mutable struct ScalingStepper{T <: Real} <: Stepper{T}
    in_affine_phase::Bool
    gamma::T

    rhs::Vector{T}
    x_rhs
    y_rhs
    z_rhs
    tau_rhs
    s_rhs
    s_rhs_k
    kap_rhs

    dir::Vector{T}
    x_dir
    y_dir
    z_dir
    z_dir_k
    tau_dir
    s_dir
    s_dir_k
    kap_dir

    res::Vector{T}
    x_res
    y_res
    z_res
    tau_res
    s_res
    kap_res

    tau_row::Int

    ScalingStepper{T}() where {T <: Real} = new{T}()
end

# create the stepper cache
function load(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs

    dim = n + p + 2q + 2
    rhs = zeros(T, dim)
    dir = zeros(T, dim)
    res = zeros(T, dim)
    stepper.rhs = rhs
    stepper.dir = dir
    stepper.res = res

    rows = 1:n
    stepper.x_rhs = view(rhs, rows)
    stepper.x_dir = view(dir, rows)
    stepper.x_res = view(res, rows)

    rows = n .+ (1:p)
    stepper.y_rhs = view(rhs, rows)
    stepper.y_dir = view(dir, rows)
    stepper.y_res = view(res, rows)

    rows = (n + p) .+ (1:q)
    stepper.z_rhs = view(rhs, rows)
    stepper.z_dir = view(dir, rows)
    stepper.z_dir_k = [view(dir, (n + p) .+ idxs_k) for idxs_k in cone_idxs]
    stepper.z_res = view(res, rows)

    tau_row = n + p + q + 1
    stepper.tau_row = tau_row
    stepper.tau_rhs = view(rhs, tau_row:tau_row)
    stepper.tau_dir = view(dir, tau_row:tau_row)
    stepper.tau_res = view(res, tau_row:tau_row)

    rows = tau_row .+ (1:q)
    stepper.s_rhs = view(rhs, rows)
    stepper.s_rhs_k = [view(rhs, tau_row .+ idxs_k) for idxs_k in cone_idxs]
    stepper.s_dir = view(dir, rows)
    stepper.s_dir_k = [view(dir, tau_row .+ idxs_k) for idxs_k in cone_idxs]
    stepper.s_res = view(res, rows)

    stepper.kap_rhs = view(rhs, dim:dim)
    stepper.kap_dir = view(dir, dim:dim)
    stepper.kap_res = view(res, dim:dim)

    return stepper
end

# TODO tune parameters for directions
function step(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point

    # update LHS of linear systems
    @timeit solver.timer "update_fact" update_fact(solver.system_solver, solver)

    # calculate affine/prediction directions
    stepper.in_affine_phase = true
    @timeit solver.timer "aff_dir" dir = get_directions(stepper, solver)

    # calculate correction factor gamma by finding distance aff_alpha for stepping in affine direction
    # TODO try using formulae for symmetric cones
    @timeit solver.timer "aff_alpha" aff_alpha = find_max_alpha_in_nbhd(
        stepper.z_dir, stepper.s_dir, stepper.tau_dir[1], stepper.kap_dir[1], solver,
        nbhd = one(T), prev_alpha = max(solver.prev_aff_alpha, T(1e-3)), min_alpha = T(1e-3))
    solver.prev_aff_alpha = aff_alpha

    gamma = (one(T) - aff_alpha)^3 # TODO allow different function (heuristic)
    solver.prev_gamma = stepper.gamma = gamma

    # calculate combined directions
    stepper.in_affine_phase = false
    sqrtmu = sqrt(solver.mu)
    Cones.load_point.(solver.model.cones, point.primal_views, sqrtmu)
    Cones.load_dual_point.(solver.model.cones, point.dual_views, sqrtmu)
    @timeit solver.timer "comb_dir" dir = get_directions(stepper, solver)

    # find distance alpha for stepping in combined direction
    @timeit solver.timer "comb_alpha" alpha = find_max_alpha_in_nbhd(
        stepper.z_dir, stepper.s_dir, stepper.tau_dir[1], stepper.kap_dir[1], solver,
        nbhd = one(T), prev_alpha = max(solver.prev_alpha, T(1e-3)), min_alpha = T(1e-3))

    if iszero(alpha)
        @warn("numerical failure: could not step in correction direction; terminating")
        solver.status = :NumericalFailure
        solver.keep_iterating = false
        return point
    end

    # step distance alpha in combined direction
    solver.prev_alpha = alpha
    @. point.x += alpha * stepper.x_dir
    @. point.y += alpha * stepper.y_dir
    @. point.z += alpha * stepper.z_dir
    @. point.s += alpha * stepper.s_dir
    solver.tau += alpha * stepper.tau_dir[1]
    solver.kap += alpha * stepper.kap_dir[1]
    calc_mu(solver)

    if solver.tau <= zero(T) || solver.kap <= zero(T) || solver.mu <= zero(T)
        @warn("numerical failure: tau is $(solver.tau), kappa is $(solver.kap), mu is $(solver.mu); terminating")
        solver.status = :NumericalFailure
        solver.keep_iterating = false
    end

    return point
end

# return affine or combined directions, depending on stepper.in_affine_phase
# TODO try to refactor the iterative refinement
function get_directions(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
    rhs = stepper.rhs
    dir = stepper.dir
    res = stepper.res
    system_solver = solver.system_solver

    @timeit solver.timer "update_rhs" update_rhs(stepper, solver) # different for affine vs combined phases
    @timeit solver.timer "solve_system" solve_system(system_solver, solver, dir, rhs)

    # use iterative refinement - note apply_LHS is different for affine vs combined phases
    dir_new = similar(res) # TODO avoid alloc
    iter_ref_steps = 3 # TODO handle, maybe change dynamically
    for i in 1:iter_ref_steps
        res = apply_LHS(stepper, solver) # modifies res
        res .-= rhs

        norm_inf = norm(res, Inf)
        @show i, norm_inf
        if norm_inf < 1000 * eps(T)
            break
        end

        @timeit solver.timer "solve_system" solve_system(system_solver, solver, dir_new, res)
        dir .-= dir_new
    end

    return dir
end

# update the 6x2 RHS matrix
function update_rhs(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
    rhs = stepper.rhs

    if stepper.in_affine_phase
        # x, y, z, tau rhs
        stepper.x_rhs .= solver.x_residual
        stepper.y_rhs .= solver.y_residual
        stepper.z_rhs .= solver.z_residual
        rhs[stepper.tau_row] = solver.kap + solver.primal_obj_t - solver.dual_obj_t

        # s rhs
        for (k, cone_k) in enumerate(solver.model.cones)
            duals_k = solver.point.dual_views[k]
            @. stepper.s_rhs_k[k] = -duals_k
        end

        # kap rhs
        rhs[end] = -solver.tau * solver.kap
    else
        # x, y, z, tau rhs
        rhs_factor = 1 - stepper.gamma
        stepper.x_rhs .*= rhs_factor
        stepper.y_rhs .*= rhs_factor
        stepper.z_rhs .*= rhs_factor
        rhs[stepper.tau_row] *= rhs_factor

        # s rhs (with Mehrotra correction for symmetric cones)
        gamma_sqrtmu = stepper.gamma * sqrt(solver.mu)
        for (k, cone_k) in enumerate(solver.model.cones)
            grad_k = Cones.grad(cone_k) # TODO store this so don't need to reload the point right before combined phase
            @. stepper.s_rhs_k[k] -= gamma_sqrtmu * grad_k # TODO mehrotra term
        end

        # kap rhs (with Mehrotra correction)
        rhs[end] += stepper.gamma * solver.mu - stepper.tau_dir[1] * stepper.kap_dir[1]
    end

    return rhs
end

# calculate residual on 6x6 linear system
# TODO make efficient / in-place
# TODO this is very similar to the CombinedStepper version - combine into one
function apply_LHS(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    tau_dir = stepper.tau_dir[1]
    kap_dir = stepper.kap_dir[1]

    # A'*y + G'*z + c*tau
    stepper.x_res .= model.A' * stepper.y_dir + model.G' * stepper.z_dir + model.c * tau_dir
    # -A*x + b*tau
    stepper.y_res .= -model.A * stepper.x_dir + model.b * tau_dir
    # -G*x + h*tau - s
    stepper.z_res .= -model.G * stepper.x_dir + model.h * tau_dir - stepper.s_dir
    # -c'*x - b'*y - h'*z - kap
    stepper.tau_res .= -model.c' * stepper.x_dir - model.b' * stepper.y_dir - model.h' * stepper.z_dir - kap_dir
    # s
    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        if Cones.use_dual(cone_k)
            # (du bar) mu*H_k*z_k + s_k
            @views Cones.hess_prod!(stepper.s_res[idxs_k], stepper.z_dir_k[k], cone_k)
            @. @views stepper.s_res[idxs_k] += stepper.s_dir_k[k]
        else
            # (pr bar) z_k + mu*H_k*s_k
            @views Cones.hess_prod!(stepper.s_res[idxs_k], stepper.s_dir_k[k], cone_k)
            @. @views stepper.s_res[idxs_k] += stepper.z_dir_k[k]
        end
    end
    # kapbar * tau + taubar * kap
    stepper.kap_res .= solver.kap * tau_dir + solver.tau * kap_dir

    return stepper.res
end
