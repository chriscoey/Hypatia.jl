#=
Copyright 2019, Chris Coey and contributors

advanced scaling point based stepping routine
=#

mutable struct ScalingStepper{T <: Real} <: Stepper{T}
    in_affine_phase::Bool
    gamma::T

    rhs::Matrix{T}
    x_rhs
    y_rhs
    z_rhs
    z_rhs_k
    tau_rhs
    s_rhs
    s_rhs_k
    kap_rhs

    dirs::Matrix{T}
    x_dirs
    y_dirs
    z_dirs
    tau_dirs
    s_dirs
    kap_dirs

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
    rhs = zeros(T, dim, 1) # TODO does this work as a vector rather than dim * 1 matrix?
    dirs = zeros(T, dim, 1)
    stepper.rhs = rhs
    stepper.dirs = dirs

    rows = 1:n
    stepper.x_rhs = view(rhs, rows, :)
    stepper.x_dirs = view(dirs, rows, :)

    rows = n .+ (1:p)
    stepper.y_rhs = view(rhs, rows, :)
    stepper.y_dirs = view(dirs, rows, :)

    rows = (n + p) .+ (1:q)
    stepper.z_rhs = view(rhs, rows, :)
    stepper.z_rhs_k = [view(rhs, (n + p) .+ idxs_k, :) for idxs_k in cone_idxs]
    stepper.z_dirs = view(dirs, rows, :)

    tau_row = n + p + q + 1
    stepper.tau_row = tau_row
    stepper.tau_rhs = view(rhs, tau_row:tau_row, :)
    stepper.tau_dirs = view(dirs, tau_row:tau_row, :)

    rows = tau_row .+ (1:q)
    stepper.s_rhs = view(rhs, rows, :)
    stepper.s_rhs_k = [view(rhs, tau_row .+ idxs_k, :) for idxs_k in cone_idxs]
    stepper.s_dirs = view(dirs, rows, :)

    stepper.kap_rhs = view(rhs, dim:dim, :)
    stepper.kap_dirs = view(dirs, dim:dim, :)

    return stepper
end

# TODO tune parameters for directions
function step(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point

    # update LHS of linear systems
    @timeit solver.timer "update_fact" update_fact(solver.system_solver, solver)

    # calculate affine/prediction directions
    stepper.in_affine_phase = true
    @timeit solver.timer "aff_dirs" dirs = get_directions(stepper, solver)

    # calculate correction factor gamma by finding distance aff_alpha for stepping in affine direction
    # TODO try using formulae for symmetric cones
    @timeit solver.timer "aff_alpha" aff_alpha = find_max_alpha_in_nbhd(
        stepper.z_dirs, stepper.s_dirs, stepper.tau_dirs[1], stepper.kap_dirs[1], solver,
        nbhd = one(T), prev_alpha = max(solver.prev_aff_alpha, T(1e-3)), min_alpha = T(1e-3))
    solver.prev_aff_alpha = aff_alpha

    gamma = (one(T) - aff_alpha)^3 # TODO allow different function (heuristic)
    solver.prev_gamma = stepper.gamma = gamma

    # calculate combined directions
    stepper.in_affine_phase = false
    @timeit solver.timer "comb_dirs" dirs = get_directions(stepper, solver)

    # find distance alpha for stepping in combined direction
    @timeit solver.timer "comb_alpha" comb_alpha = find_max_alpha_in_nbhd(
        stepper.z_dirs, stepper.s_dirs, stepper.tau_dirs[1], stepper.kap_dirs[1], solver,
        nbhd = one(T), prev_alpha = max(solver.prev_alpha, T(1e-3)), min_alpha = T(1e-3))

    if iszero(alpha)
        @warn("numerical failure: could not step in correction direction; terminating")
        solver.status = :NumericalFailure
        solver.keep_iterating = false
        return point
    end

    # step distance alpha in combined direction
    solver.prev_alpha = alpha
    @. point.x += alpha * stepper.x_dirs
    @. point.y += alpha * stepper.y_dirs
    @. point.z += alpha * stepper.z_dirs
    @. point.s += alpha * stepper.s_dirs
    solver.tau += alpha * stepper.tau_dirs[1]
    solver.kap += alpha * stepper.kap_dirs[1]
    calc_mu(solver)

    if solver.tau <= zero(T) || solver.kap <= zero(T) || solver.mu <= zero(T)
        @warn("numerical failure: tau is $(solver.tau), kappa is $(solver.kap), mu is $(solver.mu); terminating")
        solver.status = :NumericalFailure
        solver.keep_iterating = false
    end

    return point
end

# return affine or combined directions, depending on stepper.is_affine_phase
# TODO try to refactor the iterative refinement
function get_directions(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
    dirs = stepper.dirs
    system_solver = solver.system_solver

    @timeit solver.timer "update_rhs" rhs = update_rhs(stepper, solver) # different for affine vs combined phases
    @timeit solver.timer "solve_system" solve_system(system_solver, solver, dirs, rhs) # NOTE dense solve with cache destroys RHS

    # use iterative refinement - note calc_system_residual is different for affine vs combined phases
    iter_ref_steps = 3 # TODO handle, maybe change dynamically
    dirs_new = rhs
    dirs_new .= dirs # TODO avoid?
    for i in 1:iter_ref_steps
        # perform iterative refinement step
        @timeit solver.timer "calc_sys_res" res = calc_system_residual(stepper, solver) # modifies rhs
        norm_inf = norm(res, Inf)
        norm_2 = norm(res, 2)

        if norm_inf > eps(T)
            dirs_new .= zero(T)
            @timeit solver.timer "solve_system" solve_system(system_solver, solver, dirs_new, res)
            dirs_new .*= -1
            dirs_new .+= dirs
            @timeit solver.timer "calc_sys_res" res_new = calc_system_residual(stepper, solver)
            norm_inf_new = norm(res_new, Inf)
            norm_2_new = norm(res_new, 2)
            if norm_inf_new < norm_inf && norm_2_new < norm_2
                solver.verbose && @printf("used iter ref, norms: inf %9.2e to %9.2e, two %9.2e to %9.2e\n", norm_inf, norm_inf_new, norm_2, norm_2_new)
                copyto!(dirs, dirs_new)
            else
                break
            end
        end
    end

    return dirs
end

# update the 6x2 RHS matrix
function update_rhs(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
    rhs = stepper.rhs

    sqrtmu = sqrt(solver.mu)
    if stepper.is_affine_phase
        # x, y, z, tau rhs
        stepper.x_rhs .= solver.x_residual
        stepper.y_rhs .= solver.y_residual
        stepper.z_rhs .= solver.z_residual
        rhs[stepper.tau_row, 1] = solver.kap + solver.primal_obj_t - solver.dual_obj_t

        # s rhs
        for (k, cone_k) in enumerate(solver.model.cones)
            duals_k = solver.point.dual_views[k]
            grad_k = Cones.grad(cone_k)
            @. stepper.s_rhs_k[k] = -duals_k
        end

        # kap rhs
        rhs[end, 1] = -solver.tau * solver.kap
    else
        # x, y, z, tau rhs
        rhs_factor = 1 - stepper.gamma
        stepper.x_rhs .= rhs_factor * solver.x_residual
        stepper.y_rhs .= rhs_factor * solver.y_residual
        stepper.z_rhs .= rhs_factor * solver.z_residual
        rhs[stepper.tau_row, 1] = rhs_factor * (solver.kap + solver.primal_obj_t - solver.dual_obj_t)

        # s rhs
        gamma_sqrtmu = stepper.gamma * sqrtmu
        for (k, cone_k) in enumerate(solver.model.cones)
            duals_k = solver.point.dual_views[k]
            grad_k = Cones.grad(cone_k)
            @. stepper.s_rhs_k[k] = -duals_k - gamma_sqrtmu * grad_k # TODO Mehrotra correction term
        end

        # kap rhs
        rhs[end, 1] = -solver.tau * solver.kap + stepper.gamma * solver.mu # TODO Mehrotra correction term
    end

    return rhs
end

# calculate residual on 6x6 linear system
# TODO make efficient / in-place
function calc_system_residual(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model

    # LHS part
    # A'*y + G'*z + c*tau
    @timeit solver.timer "resx" res_x = model.A' * stepper.y_rhs + model.G' * stepper.z_rhs + model.c * stepper.tau_rhs
    # -A*x + b*tau
    @timeit solver.timer "resy" res_y = -model.A * stepper.x_rhs + model.b * stepper.tau_rhs
    # -G*x + h*tau - s
    @timeit solver.timer "resz" res_z = -model.G * stepper.x_rhs + model.h * stepper.tau_rhs - stepper.s_rhs
    # -c'*x - b'*y - h'*z - kap
    @timeit solver.timer "restau" res_tau = -model.c' * stepper.x_rhs - model.b' * stepper.y_rhs - model.h' * stepper.z_rhs - stepper.kap_rhs
    # s
    res_s = similar(res_z)
    @timeit solver.timer "resz" for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        if Cones.use_dual(cone_k)
            # (du bar) mu*H_k*z_k + s_k
            @views Cones.hess_prod!(res_s[idxs_k, :], stepper.z_rhs_k[k], cone_k)
            @. @views res_s[idxs_k, :] += stepper.s_rhs_k[k]
        else
            # (pr bar) z_k + mu*H_k*s_k
            @views Cones.hess_prod!(res_s[idxs_k, :], stepper.s_rhs_k[k], cone_k)
            @. @views res_s[idxs_k, :] += stepper.z_rhs_k[k]
        end
    end
    # kapbar * tau + taubar * kap
    res_kap = stepper.kap_rhs * solver.tau + stepper.tau_rhs * solver.kap

    # RHS part
    if stepper.is_affine_phase
        # x, y, z, tau rhs
        @. res_x[:, 1] -= solver.x_residual
        @. res_y[:, 1] -= solver.y_residual
        @. res_z[:, 1] -= solver.z_residual
        res_tau[1] -= solver.kap + solver.primal_obj_t - solver.dual_obj_t

        # s rhs
        @timeit solver.timer "resz" for (k, cone_k) in enumerate(model.cones)
            # srhs_k = -duals_k
            idxs_k = model.cone_idxs[k]
            duals_k = solver.point.dual_views[k]
            @. @views res_s[idxs_k, 1] += duals_k
        end

        # kap rhs
        res_kap[1] += solver.kap * solver.tau
    else
        # x, y, z, tau rhs
        rhs_factor = 1 - stepper.gamma
        @. res_x[:, 1] -= rhs_factor * solver.x_residual
        @. res_y[:, 1] -= rhs_factor * solver.y_residual
        @. res_z[:, 1] -= rhs_factor * solver.z_residual
        res_tau[1] -= rhs_factor * (solver.kap + solver.primal_obj_t - solver.dual_obj_t)

        # s rhs
        gamma_sqrtmu = stepper.gamma * sqrt(solver.mu)
        @timeit solver.timer "resz" for (k, cone_k) in enumerate(model.cones)
            # srhs_k = [-duals_k, -duals_k - mu * grad_k]
            idxs_k = model.cone_idxs[k]
            duals_k = solver.point.dual_views[k]
            grad_k = Cones.grad(cone_k)
            @. @views res_s[idxs_k, 1] += duals_k + gamma_sqrtmu * grad_k
        end

        # kap rhs
        res_kap[1] += solver.kap * solver.tau - stepper.gamma * solver.mu
    end

    return vcat(res_x, res_y, res_z, res_tau, res_s, res_kap) # TODO don't vcat
end
