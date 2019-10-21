#=
Copyright 2019, Chris Coey and contributors

combined directions stepping routine
=#

mutable struct CombinedStepper{T <: Real} <: Stepper{T}
    rhs::Matrix{T}
    x_rhs
    x_rhs1
    x_rhs2
    y_rhs
    y_rhs1
    y_rhs2
    z_rhs
    z_rhs1
    z_rhs2
    z_rhs_k
    tau_rhs
    s_rhs
    s_rhs1
    s_rhs2
    s_rhs_k
    s_rhs1_k
    s_rhs2_k
    kap_rhs

    dirs::Matrix{T}
    x_dirs
    x_pred
    x_corr
    y_dirs
    y_pred
    y_corr
    z_dirs
    z_pred
    z_corr
    tau_dirs
    s_dirs
    s_pred
    s_corr
    kap_dirs

    tau_row::Int

    CombinedStepper{T}() where {T <: Real} = new{T}()
end

# create the stepper cache
function load(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs

    dim = n + p + 2q + 2
    rhs = zeros(T, dim, 2)
    dirs = zeros(T, dim, 2)
    stepper.rhs = rhs
    stepper.dirs = dirs

    rows = 1:n
    stepper.x_rhs = view(rhs, rows, :)
    stepper.x_rhs1 = view(rhs, rows, 1)
    stepper.x_rhs2 = view(rhs, rows, 2)
    stepper.x_dirs = view(dirs, rows, :)
    stepper.x_pred = view(dirs, rows, 1)
    stepper.x_corr = view(dirs, rows, 2)

    rows = n .+ (1:p)
    stepper.y_rhs = view(rhs, rows, :)
    stepper.y_rhs1 = view(rhs, rows, 1)
    stepper.y_rhs2 = view(rhs, rows, 2)
    stepper.y_dirs = view(dirs, rows, :)
    stepper.y_pred = view(dirs, rows, 1)
    stepper.y_corr = view(dirs, rows, 2)

    rows = (n + p) .+ (1:q)
    stepper.z_rhs = view(rhs, rows, :)
    stepper.z_rhs1 = view(rhs, rows, 1)
    stepper.z_rhs2 = view(rhs, rows, 2)
    stepper.z_rhs_k = [view(rhs, (n + p) .+ idxs_k, :) for idxs_k in cone_idxs]
    stepper.z_dirs = view(dirs, rows, :)
    stepper.z_pred = view(dirs, rows, 1)
    stepper.z_corr = view(dirs, rows, 2)

    tau_row = n + p + q + 1
    stepper.tau_row = tau_row
    stepper.tau_rhs = view(rhs, tau_row:tau_row, :)
    stepper.tau_dirs = view(dirs, tau_row:tau_row, :)

    rows = tau_row .+ (1:q)
    stepper.s_rhs = view(rhs, rows, :)
    stepper.s_rhs1 = view(rhs, rows, 1)
    stepper.s_rhs2 = view(rhs, rows, 2)
    stepper.s_rhs_k = [view(rhs, tau_row .+ idxs_k, :) for idxs_k in cone_idxs]
    stepper.s_rhs1_k = [view(rhs, tau_row .+ idxs_k, 1) for idxs_k in cone_idxs]
    stepper.s_rhs2_k = [view(rhs, tau_row .+ idxs_k, 2) for idxs_k in cone_idxs]
    stepper.s_dirs = view(dirs, rows, :)
    stepper.s_pred = view(dirs, rows, 1)
    stepper.s_corr = view(dirs, rows, 2)

    stepper.kap_rhs = view(rhs, dim:dim, :)
    stepper.kap_dirs = view(dirs, dim:dim, :)

    return stepper
end

function step(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point

    # calculate affine/prediction and correction directions
    @timeit solver.timer "directions" dirs = get_directions(stepper, solver)
    (tau_pred, tau_corr, kap_pred, kap_corr) = (stepper.tau_dirs[1], stepper.tau_dirs[2], stepper.kap_dirs[1], stepper.kap_dirs[2])

    # calculate correction factor gamma by finding distance aff_alpha for stepping in affine direction
    # TODO try setting nbhd to T(Inf) and avoiding the neighborhood checks - requires tuning
    @timeit solver.timer "aff_alpha" aff_alpha = find_max_alpha_in_nbhd(
        stepper.z_pred, stepper.s_pred, tau_pred, kap_pred, solver,
        nbhd = one(T), prev_alpha = max(solver.prev_aff_alpha, T(2e-2)), min_alpha = T(2e-2))
    solver.prev_aff_alpha = aff_alpha

    gamma = abs2(one(T) - aff_alpha) # TODO allow different function (heuristic)
    solver.prev_gamma = gamma

    # find distance alpha for stepping in combined direction
    z_comb = stepper.z_pred
    s_comb = stepper.s_pred
    pred_factor = one(T) - gamma
    @. z_comb = pred_factor * stepper.z_pred + gamma * stepper.z_corr
    @. s_comb = pred_factor * stepper.s_pred + gamma * stepper.s_corr
    tau_comb = pred_factor * tau_pred + gamma * tau_corr
    kap_comb = pred_factor * kap_pred + gamma * kap_corr
    @timeit solver.timer "comb_alpha" alpha = find_max_alpha_in_nbhd(
        z_comb, s_comb, tau_comb, kap_comb, solver,
        nbhd = solver.max_nbhd, prev_alpha = solver.prev_alpha, min_alpha = T(1e-2))

    if iszero(alpha)
        # could not step far in combined direction, so perform a pure correction step
        solver.verbose && println("performing correction step")
        z_comb = stepper.z_corr
        s_comb = stepper.s_corr
        tau_comb = tau_corr
        kap_comb = kap_corr
        @timeit solver.timer "corr_alpha" alpha = find_max_alpha_in_nbhd(
            z_comb, s_comb, tau_comb, kap_comb, solver,
            nbhd = solver.max_nbhd, prev_alpha = one(T), min_alpha = T(1e-6))

        if iszero(alpha)
            @warn("numerical failure: could not step in correction direction; terminating")
            solver.status = :NumericalFailure
            solver.keep_iterating = false
            return point
        end
        @. point.x += alpha * stepper.x_corr
        @. point.y += alpha * stepper.y_corr
    else
        @. point.x += alpha * (pred_factor * stepper.x_pred + gamma * stepper.x_corr)
        @. point.y += alpha * (pred_factor * stepper.y_pred + gamma * stepper.y_corr)
    end
    solver.prev_alpha = alpha

    # step distance alpha in combined direction
    @. point.z += alpha * z_comb
    @. point.s += alpha * s_comb
    solver.tau += alpha * tau_comb
    solver.kap += alpha * kap_comb
    calc_mu(solver)

    if solver.tau <= zero(T) || solver.kap <= zero(T) || solver.mu <= zero(T)
        @warn("numerical failure: tau is $(solver.tau), kappa is $(solver.kap), mu is $(solver.mu); terminating")
        solver.status = :NumericalFailure
        solver.keep_iterating = false
    end

    return point
end

# return directions
# TODO make this function the same for all system solvers, move to solver.jl
function get_directions(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    dirs = stepper.dirs
    system_solver = solver.system_solver

    @timeit solver.timer "update_rhs" rhs = update_rhs(stepper, solver)
    @timeit solver.timer "update_fact" update_fact(system_solver, solver)
    @timeit solver.timer "solve_system" solve_system(system_solver, solver, dirs, rhs) # NOTE dense solve with cache destroys RHS

    # TODO don't do iterative refinement unless it's likely to be worthwhile based on the current worst residuals on the KKT conditions
    iter_ref_steps = 3 # TODO handle, maybe change dynamically
    dirs_new = rhs # TODO avoid by prealloc, reduce confusion
    copyto!(dirs_new, dirs)
    @timeit solver.timer "calc_sys_res" res = calc_system_residual(stepper, solver) # modifies rhs
    norm_inf = norm(res, Inf)
    norm_2 = norm(res, 2)
    for i in 1:iter_ref_steps
        if norm_inf < 100 * eps(T)
            break
        end
        # dirs_new .= zero(T) # TODO maybe want for the indirect methods
        @timeit solver.timer "solve_system" solve_system(system_solver, solver, dirs_new, res)
        @. dirs_new = dirs - dirs_new
        @timeit solver.timer "calc_sys_res" res = calc_system_residual(stepper, solver)
        norm_inf_new = norm(res, Inf)
        norm_2_new = norm(res, 2)
        if norm_inf_new > norm_inf || norm_2_new > norm_2
            break
        end
        # residual has improved, so use the iterative refinement
        solver.verbose && @printf("used iter ref, norms: inf %9.2e to %9.2e, two %9.2e to %9.2e\n", norm_inf, norm_inf_new, norm_2, norm_2_new)
        copyto!(dirs, dirs_new)
        norm_inf = norm_inf_new
        norm_2 = norm_2_new
    end

    return dirs
end

# update the 6x2 RHS matrix
function update_rhs(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
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
function calc_system_residual(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model

    # A'*y + G'*z + c*tau = [x_residual, 0]
    res_x = model.G' * stepper.z_rhs + model.c * stepper.tau_rhs
    @. res_x[:, 1] -= solver.x_residual
    # -G*x + h*tau - s = [z_residual, 0]
    res_z = -model.G * stepper.x_rhs + model.h * stepper.tau_rhs - stepper.s_rhs
    @. res_z[:, 1] -= solver.z_residual
    # -c'*x - b'*y - h'*z - kap = [kap + primal_obj_t - dual_obj_t, 0]
    res_tau = -model.c' * stepper.x_rhs - model.h' * stepper.z_rhs - stepper.kap_rhs
    res_tau[1] -= solver.kap + solver.primal_obj_t - solver.dual_obj_t

    if !iszero(model.p)
        res_x += model.A' * stepper.y_rhs
        # -A*x + b*tau = [y_residual, 0]
        res_y = -model.A * stepper.x_rhs + model.b * stepper.tau_rhs
        @. res_y[:, 1] -= solver.y_residual
        res_tau -= model.b' * stepper.y_rhs
    else
        res_y = stepper.y_rhs
    end

    sqrtmu = sqrt(solver.mu)
    res_s = similar(res_z)
    for (k, cone_k) in enumerate(model.cones)
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
