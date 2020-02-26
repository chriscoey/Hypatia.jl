#=
Copyright 2019, Chris Coey and contributors

interior point stepping routines for algorithms based on homogeneous self dual embedding
=#

mutable struct CombinedStepper{T <: Real} <: Stepper{T}
    prev_aff_alpha::T
    prev_alpha::T
    prev_gamma::T
    rhs::Vector{T}
    x_rhs
    y_rhs
    z_rhs
    s_rhs
    s_rhs_k::Vector
    dir::Vector{T}
    x_dir
    y_dir
    z_dir
    dual_dir_k::Vector
    s_dir
    primal_dir_k::Vector
    dir_temp::Vector{T}
    dir_corr::Vector{T}
    res::Vector{T}
    x_res
    y_res
    z_res
    s_res
    s_res_k::Vector
    tau_row::Int
    kap_row::Int
    z_linesearch::Vector{T}
    s_linesearch::Vector{T}
    primal_views_linesearch::Vector
    dual_views_linesearch::Vector
    CombinedStepper{T}() where {T <: Real} = new{T}()
end

# create the stepper cache
function load(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    stepper.prev_aff_alpha = one(T)
    stepper.prev_gamma = one(T)
    stepper.prev_alpha = one(T)

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
    stepper.dir_temp = zeros(T, dim)
    stepper.dir_corr = zeros(T, dim)
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
    stepper.z_res = view(res, rows)

    tau_row = n + p + q + 1
    stepper.tau_row = tau_row

    rows = tau_row .+ (1:q)
    stepper.s_rhs = view(rhs, rows)
    stepper.s_rhs_k = [view(rhs, tau_row .+ idxs_k) for idxs_k in cone_idxs]
    stepper.s_dir = view(dir, rows)
    stepper.s_res = view(res, rows)
    stepper.s_res_k = [view(res, tau_row .+ idxs_k) for idxs_k in cone_idxs]

    stepper.primal_dir_k = similar(stepper.s_res_k)
    stepper.dual_dir_k = similar(stepper.s_res_k)
    for (k, idxs_k) in enumerate(cone_idxs)
        s_k = view(dir, tau_row .+ idxs_k)
        z_k = view(dir, (n + p) .+ idxs_k)
        (stepper.primal_dir_k[k], stepper.dual_dir_k[k]) = (Cones.use_dual_barrier(cones[k]) ? (z_k, s_k) : (s_k, z_k))
    end

    stepper.kap_row = dim

    stepper.z_linesearch = zeros(T, q)
    stepper.s_linesearch = zeros(T, q)
    stepper.primal_views_linesearch = [view(Cones.use_dual_barrier(model.cones[k]) ? stepper.z_linesearch : stepper.s_linesearch, model.cone_idxs[k]) for k in eachindex(model.cones)]
    stepper.dual_views_linesearch = [view(Cones.use_dual_barrier(model.cones[k]) ? stepper.s_linesearch : stepper.z_linesearch, model.cone_idxs[k]) for k in eachindex(model.cones)]

    return stepper
end

function step(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point

    # update linear system solver factorization and helpers
    @timeit solver.timer "update_fact" update_fact(solver.system_solver, solver)

    # calculate correction direction and keep in dir_corr
    @timeit solver.timer "rhs_corr" update_rhs_correction(stepper, solver)
    @timeit solver.timer "corr_dir" get_directions(stepper, solver, iter_ref_steps = 3)
    copyto!(stepper.dir_corr, stepper.dir)

    # calculate affine/prediction direction and keep in dir
    @timeit solver.timer "rhs_aff" update_rhs_affine(stepper, solver)
    @timeit solver.timer "aff_dir" get_directions(stepper, solver, iter_ref_steps = 3)

    # calculate correction factor gamma by finding distance aff_alpha for stepping in affine direction
    @timeit solver.timer "aff_alpha" stepper.prev_aff_alpha = aff_alpha = find_max_alpha(
        stepper, solver, prev_alpha = stepper.prev_aff_alpha, min_alpha = T(1e-2))
    stepper.prev_gamma = gamma = abs2(one(T) - aff_alpha) # TODO allow different function (heuristic) as option?

    # calculate combined direction and keep in dir
    axpby!(gamma, stepper.dir_corr, 1 - gamma, stepper.dir)

    # find distance alpha for stepping in combined direction
    @timeit solver.timer "comb_alpha" alpha = find_max_alpha(
        stepper, solver, prev_alpha = stepper.prev_alpha, min_alpha = T(1e-3))

    if iszero(alpha)
        # could not step far in combined direction, so attempt a pure correction step
        solver.verbose && println("performing correction step")
        copyto!(stepper.dir, stepper.dir_corr)

        # find distance alpha for stepping in correction direction
        @timeit solver.timer "corr_alpha" alpha = find_max_alpha(
            stepper, solver, prev_alpha = one(T), min_alpha = T(1e-6))

        if iszero(alpha)
            @warn("numerical failure: could not step in correction direction; terminating")
            solver.status = :NumericalFailure
            return false
        end
    end
    stepper.prev_alpha = alpha

    # step distance alpha in combined direction
    @. point.x += alpha * stepper.x_dir
    @. point.y += alpha * stepper.y_dir
    @. point.z += alpha * stepper.z_dir
    @. point.s += alpha * stepper.s_dir
    solver.tau += alpha * stepper.dir[stepper.tau_row]
    solver.kap += alpha * stepper.dir[stepper.kap_row]
    calc_mu(solver)

    if solver.tau <= zero(T) || solver.kap <= zero(T) || solver.mu <= zero(T)
        @warn("numerical failure: tau is $(solver.tau), kappa is $(solver.kap), mu is $(solver.mu); terminating")
        solver.status = :NumericalFailure
        return false
    end

    return true # step succeeded
end

# update the RHS for affine direction
function update_rhs_affine(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    rhs = stepper.rhs

    # x, y, z, tau
    stepper.x_rhs .= solver.x_residual
    stepper.y_rhs .= solver.y_residual
    stepper.z_rhs .= solver.z_residual
    rhs[stepper.tau_row] = solver.kap + solver.primal_obj_t - solver.dual_obj_t

    # s
    for k in eachindex(solver.model.cones)
        duals_k = solver.point.dual_views[k]
        @. stepper.s_rhs_k[k] = -duals_k
    end

    # kap
    rhs[end] = -solver.kap
    # TODO NT: -solver.tau

    return rhs
end

# update the RHS for correction direction
function update_rhs_correction(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    rhs = stepper.rhs

    # x, y, z, tau
    stepper.rhs[1:stepper.tau_row] .= 0

    # s
    for (k, cone_k) in enumerate(solver.model.cones)
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cone_k)
        @. stepper.s_rhs_k[k] = -duals_k - solver.mu * grad_k
        # if Cones.use_3order_corr(cone_k)
        #     # TODO check math here for case of cone.use_dual true - should s and z be swapped then?
        #     stepper.s_rhs_k[k] .-= Cones.correction(cone_k, stepper.primal_dir_k[k], stepper.dual_dir_k[k])
        # end
    end

    # kap
    rhs[end] = -solver.kap + solver.mu / solver.tau

    return rhs
end

# calculate direction given rhs, and apply iterative refinement
function get_directions(stepper::CombinedStepper{T}, solver::Solver{T}; iter_ref_steps::Int = 0) where {T <: Real}
    rhs = stepper.rhs
    dir = stepper.dir
    dir_temp = stepper.dir_temp
    res = stepper.res
    system_solver = solver.system_solver

    @timeit solver.timer "solve_system" solve_system(system_solver, solver, dir, rhs)

    # use iterative refinement
    copyto!(dir_temp, dir)
    res = apply_lhs(stepper, solver) # modifies res
    res .-= rhs
    norm_inf = norm(res, Inf)
    norm_2 = norm(res, 2)

    for i in 1:iter_ref_steps
        if norm_inf < 100 * eps(T) # TODO change tolerance dynamically
            break
        end
        @timeit solver.timer "solve_system" solve_system(system_solver, solver, dir, res)
        axpby!(true, dir_temp, -1, dir)
        res = apply_lhs(stepper, solver) # modifies res
        res .-= rhs

        norm_inf_new = norm(res, Inf)
        norm_2_new = norm(res, 2)
        if norm_inf_new > norm_inf || norm_2_new > norm_2
            # residual has not improved
            copyto!(dir, dir_temp)
            break
        end

        # residual has improved, so use the iterative refinement
        # TODO only print if using debug mode
        # solver.verbose && @printf("iter ref round %d norms: inf %9.2e to %9.2e, two %9.2e to %9.2e\n", i, norm_inf, norm_inf_new, norm_2, norm_2_new)
        copyto!(dir_temp, dir)
        norm_inf = norm_inf_new
        norm_2 = norm_2_new
    end

    return dir
end

# calculate residual on 6x6 linear system
function apply_lhs(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    tau_dir = stepper.dir[stepper.tau_row]
    kap_dir = stepper.dir[stepper.kap_row]

    # A'*y + G'*z + c*tau
    copyto!(stepper.x_res, model.c)
    mul!(stepper.x_res, model.G', stepper.z_dir, true, tau_dir)
    # -G*x + h*tau - s
    @. stepper.z_res = model.h * tau_dir - stepper.s_dir
    mul!(stepper.z_res, model.G, stepper.x_dir, -1, true)
    # -c'*x - b'*y - h'*z - kap
    stepper.res[stepper.tau_row] = -dot(model.c, stepper.x_dir) - dot(model.h, stepper.z_dir) - kap_dir
    # if p = 0, ignore A, b, y
    if !iszero(model.p)
        # A'*y + G'*z + c*tau
        mul!(stepper.x_res, model.A', stepper.y_dir, true, true)
        # -A*x + b*tau
        copyto!(stepper.y_res, model.b)
        mul!(stepper.y_res, model.A, stepper.x_dir, -1, tau_dir)
        # -c'*x - b'*y - h'*z - kap
        stepper.res[stepper.tau_row] -= dot(model.b, stepper.y_dir)
    end

    # s
    for (k, cone_k) in enumerate(model.cones)
        # (du bar) mu*H_k*z_k + s_k
        # (pr bar) z_k + mu*H_k*s_k
        s_res_k = stepper.s_res_k[k]
        Cones.hess_prod!(s_res_k, stepper.primal_dir_k[k], cone_k)
        lmul!(solver.mu, s_res_k)
        @. s_res_k += stepper.dual_dir_k[k]
    end

    # mu / (taubar^2) * tau + kap
    stepper.res[stepper.kap_row] = solver.mu / solver.tau * tau_dir / solver.tau + kap_dir
    # TODO NT: tau + taubar / kapbar * kap
    # stepper.res[stepper.kap_row] = tau_dir + solver.tau / solver.kap * kap_dir

    return stepper.res
end

# backtracking line search to find large distance to step in direction while remaining inside cones and inside a given neighborhood
function find_max_alpha(
    stepper::CombinedStepper{T},
    solver::Solver{T};
    prev_alpha::T,
    min_alpha::T,
    ) where {T <: Real}
    z = solver.point.z
    s = solver.point.s
    tau = solver.tau
    kap = solver.kap
    z_dir = stepper.z_dir
    s_dir = stepper.s_dir
    tau_dir = stepper.dir[stepper.tau_row]
    kap_dir = stepper.dir[stepper.kap_row]
    z_linesearch = stepper.z_linesearch
    s_linesearch = stepper.s_linesearch

    alpha = max(T(0.1), min(prev_alpha * T(1.4), one(T))) # TODO option for parameter
    if tau_dir < zero(T)
        alpha = min(alpha, -tau / tau_dir)
    end
    if kap_dir < zero(T)
        alpha = min(alpha, -kap / kap_dir)
    end
    alpha *= T(0.9999)

    tau_temp = kap_temp = taukap_temp = mu_temp = zero(T)
    nup1 = solver.model.nu + 1
    while true
        @. z_linesearch = z + alpha * z_dir
        @. s_linesearch = s + alpha * s_dir
        tau_temp = tau + alpha * tau_dir
        kap_temp = kap + alpha * kap_dir
        taukap_temp = tau_temp * kap_temp
        mu_temp = (dot(s_linesearch, z_linesearch) + taukap_temp) / nup1

        if mu_temp > eps(T) && abs(taukap_temp - mu_temp) < mu_temp * solver.max_nbhd
            # TODO order the cones by how long it takes to check these conditions and iterate in that order, to improve efficiency
            in_nbhd = true
            for (k, cone_k) in enumerate(solver.model.cones)
                Cones.load_point(cone_k, stepper.primal_views_linesearch[k])
                Cones.reset_data(cone_k)
                if !Cones.is_feas(cone_k) || !Cones.in_neighborhood(cone_k, stepper.dual_views_linesearch[k], mu_temp)
                    in_nbhd = false
                    break
                end
            end
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
        alpha *= T(0.9) # TODO option for parameter
    end

    return alpha
end

# TODO if p = 0, don't print y_feas
function print_iteration_stats(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    if iszero(solver.num_iters)
        @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
            "iter", "p_obj", "d_obj", "abs_gap", "rel_gap",
            "x_feas", "y_feas", "z_feas", "tau", "kap", "mu",
            "gamma", "alpha",
            )
        @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
            solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap, solver.rel_gap,
            solver.x_feas, solver.y_feas, solver.z_feas, solver.tau, solver.kap, solver.mu
            )
    else
        @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
            solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap, solver.rel_gap,
            solver.x_feas, solver.y_feas, solver.z_feas, solver.tau, solver.kap, solver.mu,
            stepper.prev_gamma, stepper.prev_alpha,
            )
    end
    flush(stdout)
    return
end

# TODO experimental for BlockMatrix LHS: if block is a Cone then define mul as hessian product, if block is solver then define mul by mu/tau/tau
# TODO optimize... maybe need for each cone a 5-arg hess prod
# import LinearAlgebra.mul!
#
# function mul!(y::AbstractVecOrMat{T}, A::Cones.Cone{T}, x::AbstractVecOrMat{T}, alpha::Number, beta::Number) where {T <: Real}
#     # TODO in-place
#     ytemp = y * beta
#     Cones.hess_prod!(y, x, A)
#     rmul!(y, alpha)
#     y .+= ytemp
#     return y
# end
#
# function mul!(y::AbstractVecOrMat{T}, solver::Solvers.Solver{T}, x::AbstractVecOrMat{T}, alpha::Number, beta::Number) where {T <: Real}
#     rmul!(y, beta)
#     @. y += alpha * x / solver.tau * solver.mu / solver.tau
#     return y
# end
