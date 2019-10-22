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
    s_res_k
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
    stepper.s_res_k = [view(res, tau_row .+ idxs_k) for idxs_k in cone_idxs]

    stepper.kap_rhs = view(rhs, dim:dim)
    stepper.kap_dir = view(dir, dim:dim)
    stepper.kap_res = view(res, dim:dim)

    return stepper
end

# TODO tune parameters for directions
function step(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point

    @timeit solver.timer "update_fact" update_fact(solver.system_solver, solver)

    # calculate affine/prediction directions
    stepper.in_affine_phase = true
    @timeit solver.timer "aff_dir" dir = get_directions(stepper, solver)

    # calculate correction factor gamma by finding distance aff_alpha for stepping in affine direction
    solver.prev_aff_alpha = aff_alpha = find_max_alpha(stepper.z_dir, stepper.s_dir, stepper.tau_dir[1], stepper.kap_dir[1], solver)
    # @timeit solver.timer "aff_alpha" solver.prev_aff_alpha = aff_alpha = find_max_alpha_in_nbhd(
    #     stepper.z_dir, stepper.s_dir, stepper.tau_dir[1], stepper.kap_dir[1], solver,
    #     nbhd = one(T), prev_alpha = max(solver.prev_aff_alpha, T(2e-2)), min_alpha = T(2e-2))
    # @show aff_alpha

    @assert 0 <= aff_alpha <= 1
    gamma = (1 - aff_alpha)^3 # TODO allow different function (heuristic) # TODO rename gamma to sigma maybe, if get rid of combined stepper
    solver.prev_gamma = stepper.gamma = gamma
    @show gamma

    Cones.load_point.(solver.model.cones, point.primal_views)
    Cones.load_dual_point.(solver.model.cones, point.dual_views)
    Cones.reset_data.(solver.model.cones)
    Cones.is_feas.(solver.model.cones)
    Cones.grad.(solver.model.cones)

    # calculate combined directions
    stepper.in_affine_phase = false
    @timeit solver.timer "comb_dir" dir = get_directions(stepper, solver)

    # find distance alpha for stepping in combined direction
    alpha = 0.99999 * find_max_alpha(stepper.z_dir, stepper.s_dir, stepper.tau_dir[1], stepper.kap_dir[1], solver)
    # @timeit solver.timer "alpha" solver.prev_alpha = alpha = find_max_alpha_in_nbhd(
    #     stepper.z_dir, stepper.s_dir, stepper.tau_dir[1], stepper.kap_dir[1], solver,
    #     nbhd = solver.max_nbhd, prev_alpha = solver.prev_alpha, min_alpha = T(1e-2))
    # @show alpha

    @assert 0 < alpha <= 1

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

    Cones.load_point.(solver.model.cones, point.primal_views)
    Cones.load_dual_point.(solver.model.cones, point.dual_views)
    Cones.reset_data.(solver.model.cones)
    Cones.is_feas.(solver.model.cones)
    Cones.grad.(solver.model.cones)

    return point
end

function find_max_alpha(
    z_dir::AbstractVector{T},
    s_dir::AbstractVector{T},
    tau_dir::T,
    kap_dir::T,
    solver::Solver{T},
    ) where {T <: Real}
    alpha = one(T)

    if kap_dir < zero(T)
        alpha = min(alpha, -solver.kap / kap_dir)
    end
    if tau_dir < zero(T)
        alpha = min(alpha, -solver.tau / tau_dir)
    end

    for (cone_k, idxs_k) in zip(solver.model.cones, solver.model.cone_idxs)
        if Cones.use_scaling(cone_k)
            @views dist_k = Cones.step_max_dist(cone_k, s_dir[idxs_k], z_dir[idxs_k])
        else
            # TODO
        end
        alpha = min(alpha, dist_k)
    end
    # alpha = minimum(Cones.step_max_dist(cone_k, s_dir[idxs_k], z_dir[idxs_k]) for (cone_k, idxs_k) in zip(solver.model.cones, solver.model.cone_idxs))

    return alpha
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
        if norm_inf < 1000 * eps(T) # TODO also stop if residual not getting better
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
        for k in eachindex(solver.model.cones)
            duals_k = solver.point.dual_views[k]
            @. stepper.s_rhs_k[k] = -duals_k
        end

        # kap rhs
        rhs[end] = -solver.tau
    else
        # x, y, z, tau rhs
        rhs_factor = 1 - stepper.gamma
        lmul!(rhs_factor, stepper.x_rhs)
        lmul!(rhs_factor, stepper.y_rhs)
        lmul!(rhs_factor, stepper.z_rhs)
        rhs[stepper.tau_row] *= rhs_factor

        gamma_mu = stepper.gamma * solver.mu

        # s rhs (with Mehrotra correction for symmetric cones)
        for (k, cone_k) in enumerate(solver.model.cones)
            # TODO store this if doing line search so don't need to reload the point right before combined phase
            grad_k = Cones.grad(cone_k)
            @. stepper.s_rhs_k[k] -= gamma_mu * grad_k
            if Cones.use_scaling(cone_k)
                corr = Cones.correction(cone_k, stepper.s_dir_k[k], stepper.z_dir_k[k])
                @. stepper.s_rhs_k[k] -= corr
            end
        end

        # kap rhs (with Mehrotra correction)
        rhs[end] += (gamma_mu - stepper.tau_dir[1] * stepper.kap_dir[1]) / solver.kap
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

    # TODO ignore A, b, y if p = 0
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
        if Cones.use_dual(cone_k)
            # (du bar) mu*H_k*z_k + s_k
            Hzs_k = stepper.z_dir_k[k]
            zs_k = stepper.s_dir_k[k]
        else
            # (pr bar) z_k + mu*H_k*s_k
            Hzs_k = stepper.s_dir_k[k]
            zs_k = stepper.z_dir_k[k]
        end
        s_res_k = stepper.s_res_k[k]
        Cones.hess_prod!(s_res_k, Hzs_k, cone_k)
        if !Cones.use_scaling(cone_k)
            lmul!(solver.mu, s_res_k)
        end
        @. s_res_k += zs_k
    end

    # tau + taubar / kapbar * kap
    @. stepper.kap_res = tau_dir + solver.tau / solver.kap * kap_dir

    return stepper.res
end
