#=
Copyright 2019, Chris Coey and contributors

advanced scaling point based stepping routine
=#

mutable struct ScalingStepper{T <: Real} <: Stepper{T}
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

    # calculate affine/prediction directions
    @timeit solver.timer "aff_dirs" dirs = get_affine_directions(stepper, solver)
    (tau_pred, kap_pred) = (stepper.tau_dirs[1], stepper.kap_dirs[1])

    # calculate correction factor gamma by finding distance aff_alpha for stepping in affine direction
    # TODO try using formulae for symmetric cones
    @timeit solver.timer "aff_alpha" aff_alpha = find_max_alpha_in_nbhd(
        stepper.z_dirs, stepper.s_dirs, stepper.tau_dirs[1], stepper.kap_dirs[1], solver,
        nbhd = one(T), prev_alpha = max(solver.prev_aff_alpha, T(1e-3)), min_alpha = T(1e-3))
    solver.prev_aff_alpha = aff_alpha

    gamma = (one(T) - aff_alpha)^3 # TODO allow different function (heuristic)
    solver.prev_gamma = gamma

    # calculate combined directions
    @timeit solver.timer "comb_dirs" dirs = get_combined_directions(stepper, solver)
    (tau_comb, kap_comb) = (stepper.tau_dirs[1], stepper.kap_dirs[1])

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
    @. point.x += alpha * stepper.x_corr
    @. point.y += alpha * stepper.y_corr
    @. point.z += alpha * z_comb
    @. point.s += alpha * s_comb
    solver.tau += alpha * tau_comb
    solver.kap += alpha * kap_comb
    calc_mu(solver)
    solver.prev_alpha = alpha

    if solver.tau <= zero(T) || solver.kap <= zero(T) || solver.mu <= zero(T)
        @warn("numerical failure: tau is $(solver.tau), kappa is $(solver.kap), mu is $(solver.mu); terminating")
        solver.status = :NumericalFailure
        solver.keep_iterating = false
    end

    return point
end

# return affine directions
# TODO try to refactor the iterative refinement
function get_affine_directions(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
    dirs = stepper.dirs
    system_solver = solver.system_solver

    @timeit solver.timer "update_fact" update_fact(system_solver, solver)

    @timeit solver.timer "update_rhs" rhs = update_rhs(stepper, solver)
    @timeit solver.timer "solve_system" solve_system(system_solver, solver, dirs, rhs) # NOTE dense solve with cache destroys RHS

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

# return combined directions
# TODO try to refactor the iterative refinement
function get_combined_directions(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
    dirs = stepper.dirs
    system_solver = solver.system_solver

    @timeit solver.timer "update_rhs" rhs = update_rhs(stepper, solver)
    @timeit solver.timer "solve_system" solve_system(system_solver, solver, dirs, rhs) # NOTE dense solve with cache destroys RHS

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

    stepper.x_rhs1 .= solver.x_residual
    stepper.x_rhs2 .= zero(T)
    stepper.y_rhs1 .= solver.y_residual
    stepper.y_rhs2 .= zero(T)
    stepper.z_rhs1 .= solver.z_residual
    stepper.z_rhs2 .= zero(T)
    rhs[stepper.tau_row, 1] = solver.kap + solver.primal_obj_t - solver.dual_obj_t
    rhs[stepper.tau_row, 2] = zero(T)

    sqrtmu = sqrt(solver.mu)
    for (k, cone_k) in enumerate(solver.model.cones)
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cone_k)
        @. stepper.s_rhs1_k[k] = -duals_k
        @. stepper.s_rhs2_k[k] = -duals_k - grad_k * sqrtmu
    end

    rhs[end, 1] = -solver.kap
    rhs[end, 2] = -solver.kap + solver.mu / solver.tau

    return rhs
end

# calculate residual on 6x6 linear system
# TODO make efficient / in-place
function calc_system_residual(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model

    # A'*y + G'*z + c*tau = [x_residual, 0]
    @timeit solver.timer "resx" res_x = model.A' * stepper.y_rhs + model.G' * stepper.z_rhs + model.c * stepper.tau_rhs
    @. res_x[:, 1] -= solver.x_residual
    # -A*x + b*tau = [y_residual, 0]
    @timeit solver.timer "resy" res_y = -model.A * stepper.x_rhs + model.b * stepper.tau_rhs
    @. res_y[:, 1] -= solver.y_residual
    # -G*x + h*tau - s = [z_residual, 0]
    @timeit solver.timer "resz" res_z = -model.G * stepper.x_rhs + model.h * stepper.tau_rhs - stepper.s_rhs
    @. res_z[:, 1] -= solver.z_residual
    # -c'*x - b'*y - h'*z - kap = [kap + primal_obj_t - dual_obj_t, 0]
    @timeit solver.timer "restau" res_tau = -model.c' * stepper.x_rhs - model.b' * stepper.y_rhs - model.h' * stepper.z_rhs - stepper.kap_rhs
    res_tau[1] -= solver.kap + solver.primal_obj_t - solver.dual_obj_t

    sqrtmu = sqrt(solver.mu)
    res_s = similar(res_z)
    @timeit solver.timer "resz" for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        if Cones.use_dual(cone_k)
            # (du bar) mu*H_k*z_k + s_k = srhs_k
            @views Cones.hess_prod!(res_s[idxs_k, :], stepper.z_rhs_k[k], cone_k)
            @. @views res_s[idxs_k, :] += stepper.s_rhs_k[k]
        else
            # (pr bar) z_k + mu*H_k*s_k = srhs_k
            @views Cones.hess_prod!(res_s[idxs_k, :], stepper.s_rhs_k[k], cone_k)
            @. @views res_s[idxs_k, :] += stepper.z_rhs_k[k]
        end
        # srhs_k = [-duals_k, -duals_k - mu * grad_k]
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cone_k)
        @. @views res_s[idxs_k, 1] += duals_k
        @. @views res_s[idxs_k, 2] += duals_k + grad_k * sqrtmu
    end

    # mu/(taubar^2)*tau + kap = [-kap, -kap + mu/tau]
    res_kap = stepper.kap_rhs + solver.mu / solver.tau * stepper.tau_rhs / solver.tau
    res_kap[1] += solver.kap
    res_kap[2] += solver.kap - solver.mu / solver.tau

    return vcat(res_x, res_y, res_z, res_tau, res_s, res_kap) # TODO don't vcat
end
