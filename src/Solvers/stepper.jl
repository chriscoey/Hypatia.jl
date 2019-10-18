#=
Copyright 2019, Chris Coey and contributors

interior point stepping routines for algorithms based on homogeneous self dual embedding
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

# return directions
# TODO make this function the same for all system solvers, move to solver.jl
function get_directions(stepper::Stepper{T}, solver::Solver{T}) where {T <: Real}
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

# backtracking line search to find large distance to step in direction while remaining inside cones and inside a given neighborhood
function find_max_alpha_in_nbhd(
    z_dir::AbstractVector{T},
    s_dir::AbstractVector{T},
    tau_dir::T,
    kap_dir::T,
    solver::Solver{T};
    nbhd::T,
    prev_alpha::T,
    min_alpha::T,
    ) where {T <: Real}
    point = solver.point
    model = solver.model
    z_temp = solver.z_temp
    s_temp = solver.s_temp

    alpha = min(prev_alpha * T(1.4), one(T)) # TODO option for parameter
    if kap_dir < zero(T)
        alpha = min(alpha, -solver.kap / kap_dir)
    end
    if tau_dir < zero(T)
        alpha = min(alpha, -solver.tau / tau_dir)
    end
    alpha *= T(0.9999)

    solver.cones_infeas .= true
    tau_temp = kap_temp = taukap_temp = mu_temp = zero(T)
    while true
        @timeit solver.timer "ls_update" begin
        @. z_temp = point.z + alpha * z_dir
        @. s_temp = point.s + alpha * s_dir
        tau_temp = solver.tau + alpha * tau_dir
        kap_temp = solver.kap + alpha * kap_dir
        taukap_temp = tau_temp * kap_temp
        mu_temp = (dot(s_temp, z_temp) + taukap_temp) / (one(T) + model.nu)
        end

        if mu_temp > zero(T)
            @timeit solver.timer "nbhd_check" in_nbhd = check_nbhd(mu_temp, taukap_temp, nbhd, solver)
            if in_nbhd
                break
            end
        end

        if alpha < min_alpha
            # alpha is very small so finish
            alpha = zero(T)
            break
        end

        # iterate is outside the neighborhood: decrease alpha
        alpha *= T(0.8) # TODO option for parameter
    end

    return alpha
end

function check_nbhd(
    mu_temp::T,
    taukap_temp::T,
    nbhd::T,
    solver::Solver{T},
    ) where {T <: Real}
    cones = solver.model.cones
    sqrtmu = sqrt(mu_temp)

    rhs_nbhd = mu_temp * abs2(nbhd)
    lhs_nbhd = abs2(taukap_temp / sqrtmu - sqrtmu)
    if lhs_nbhd >= rhs_nbhd
        return false
    end

    Cones.load_point.(cones, solver.primal_views, sqrtmu)
    Cones.load_dual_point.(cones, solver.dual_views, sqrtmu) # TODO needed?

    # accept primal iterate if it is inside the cone and neighborhood
    # first check inside cone for whichever cones were violated last line search iteration
    for (k, cone_k) in enumerate(cones)
        if solver.cones_infeas[k]
            Cones.reset_data(cone_k)
            if Cones.is_feas(cone_k)
                solver.cones_infeas[k] = false
                solver.cones_loaded[k] = true
            else
                return false
            end
        else
            solver.cones_loaded[k] = false
        end
    end

    for (k, cone_k) in enumerate(cones)
        if !solver.cones_loaded[k]
            Cones.reset_data(cone_k)
            if !Cones.is_feas(cone_k)
                return false
            end
        end

        # modifies dual_views
        duals_k = solver.dual_views[k]
        g_k = Cones.grad(cone_k)
        @. duals_k += g_k * sqrtmu

        if solver.use_infty_nbhd
            k_nbhd = abs2(norm(duals_k, Inf) / norm(g_k, Inf))
            # k_nbhd = abs2(maximum(abs(dj) / abs(gj) for (dj, gj) in zip(duals_k, g_k))) # TODO try this neighborhood
            lhs_nbhd = max(lhs_nbhd, k_nbhd)
        else
            nbhd_temp_k = solver.nbhd_temp[k]
            Cones.inv_hess_prod!(nbhd_temp_k, duals_k, cone_k)
            k_nbhd = dot(duals_k, nbhd_temp_k)
            if k_nbhd <= -cbrt(eps(T))
                @warn("numerical failure: cone neighborhood is $k_nbhd")
                return false
            elseif k_nbhd > zero(T)
                lhs_nbhd += k_nbhd
            end
        end

        if lhs_nbhd > rhs_nbhd
            return false
        end
    end

    return true
end

# TODO experimental for BlockMatrix LHS: if block is a Cone then define mul as hessian product, if block is solver then define mul by mu/tau/tau
# TODO optimize... maybe need for each cone a 5-arg hess prod
import LinearAlgebra.mul!

function mul!(y::AbstractVecOrMat{T}, A::Cones.Cone{T}, x::AbstractVecOrMat{T}, alpha::Number, beta::Number) where {T <: Real}
    # TODO in-place
    ytemp = y * beta
    Cones.hess_prod!(y, x, A)
    rmul!(y, alpha)
    y .+= ytemp
    return y
end

function mul!(y::AbstractVecOrMat{T}, solver::Solvers.Solver{T}, x::AbstractVecOrMat{T}, alpha::Number, beta::Number) where {T <: Real}
    rmul!(y, beta)
    @. y += alpha * x / solver.tau * solver.mu / solver.tau
    return y
end
