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
    s_rhs
    s_rhs_k

    dir::Vector{T}
    x_dir
    y_dir
    z_dir
    z_dir_k
    s_dir
    s_dir_k

    dir_temp::Vector{T}

    res::Vector{T}
    x_res
    y_res
    z_res
    s_res
    s_res_k

    tau_row::Int
    kap_row::Int

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
    stepper.dir_temp = zeros(T, dim)
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

    rows = tau_row .+ (1:q)
    stepper.s_rhs = view(rhs, rows)
    stepper.s_rhs_k = [view(rhs, tau_row .+ idxs_k) for idxs_k in cone_idxs]
    stepper.s_dir = view(dir, rows)
    stepper.s_dir_k = [view(dir, tau_row .+ idxs_k) for idxs_k in cone_idxs]
    stepper.s_res = view(res, rows)
    stepper.s_res_k = [view(res, tau_row .+ idxs_k) for idxs_k in cone_idxs]

    stepper.kap_row = dim

    return stepper
end

# TODO tune parameters for directions
function step(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point

    # TODO not needed for cones with last point already loaded
    Cones.load_point.(solver.model.cones, point.primal_views)
    Cones.load_dual_point.(solver.model.cones, point.dual_views)
    Cones.reset_data.(solver.model.cones)
    Cones.is_feas.(solver.model.cones)
    Cones.grad.(solver.model.cones)

    @timeit solver.timer "update_fact" update_fact(solver.system_solver, solver)

    # calculate affine/prediction directions
    stepper.in_affine_phase = true
    @timeit solver.timer "aff_dir" get_directions(stepper, solver)

    # calculate correction factor gamma by finding distance aff_alpha for stepping in affine direction
    # TODO rename gamma to sigma maybe, if get rid of combined stepper
    solver.prev_aff_alpha = aff_alpha = find_max_alpha(stepper, solver)
    @assert 0 <= aff_alpha <= 1
    gamma = (1 - aff_alpha) ^ 3 # TODO allow different function (heuristic)
    solver.prev_gamma = stepper.gamma = gamma

    # TODO needed?
    Cones.load_point.(solver.model.cones, point.primal_views)
    Cones.load_dual_point.(solver.model.cones, point.dual_views)
    Cones.reset_data.(solver.model.cones)
    Cones.is_feas.(solver.model.cones)
    Cones.grad.(solver.model.cones)

    # calculate combined directions
    stepper.in_affine_phase = false
    @timeit solver.timer "comb_dir" get_directions(stepper, solver)

    # find distance alpha for stepping in combined direction
    solver.prev_alpha = alpha = 0.99 * find_max_alpha(stepper, solver) # TODO make the constant an option, depends on eps(T)?
    if iszero(alpha)
        @warn("numerical failure: could not step in combined direction; terminating")
        solver.status = :NumericalFailure
        solver.keep_iterating = false
        return point
    end

    # step distance alpha in combined directions
    # TODO allow stepping different alphas in primal and dual cone directions? some solvers do
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
        solver.keep_iterating = false
    end

    return point
end

function find_max_alpha(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
    tau_dir = stepper.dir[stepper.tau_row]
    kap_dir = stepper.dir[stepper.kap_row]
    alpha = one(T) # alpha at most 1

    # tau and kappa
    if tau_dir < zero(T)
        alpha = min(alpha, -solver.tau / tau_dir)
    end
    if kap_dir < zero(T)
        alpha = min(alpha, -solver.kap / kap_dir)
    end

    # cones using scaling and max distance functions
    for (k, cone_k) in enumerate(solver.model.cones)
        if Cones.use_scaling(cone_k)
            dist_k = Cones.step_max_dist(cone_k, stepper.s_dir_k[k], stepper.z_dir_k[k])
            alpha = min(alpha, dist_k)
        end
    end

    if all(Cones.use_scaling, solver.model.cones)
        return alpha
    end

    # cones requiring line search and neighborhood
    nbhd = (stepper.in_affine_phase ? one(T) : solver.max_nbhd)
    point = solver.point
    model = solver.model
    z_temp = solver.z_temp
    s_temp = solver.s_temp
    solver.cones_infeas .= true
    tau_temp = kap_temp = taukap_temp = mu_temp = zero(T)
    while true
        # TODO only do nbhd checks for cones not using scaling
        @. z_temp = point.z + alpha * stepper.z_dir
        @. s_temp = point.s + alpha * stepper.s_dir
        tau_temp = solver.tau + alpha * tau_dir
        kap_temp = solver.kap + alpha * kap_dir
        taukap_temp = tau_temp * kap_temp
        mu_temp = (dot(s_temp, z_temp) + taukap_temp) / (one(T) + model.nu)

        if mu_temp > zero(T)
            @timeit solver.timer "nbhd_check" in_nbhd = check_nbhd(mu_temp, taukap_temp, nbhd, solver)
            if in_nbhd
                break
            end
        end

        if alpha < 1e-3 # TODO arg
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

    Cones.load_point.(cones, solver.primal_views)
    Cones.load_dual_point.(cones, solver.dual_views)

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

    @assert solver.use_infty_nbhd # TODO hess nbhd?
    rhs_nbhd = mu_temp * abs2(nbhd)
    for (k, cone_k) in enumerate(cones)
        if Cones.use_scaling(cone_k)
            continue
        end

        if !solver.cones_loaded[k]
            Cones.reset_data(cone_k)
            if !Cones.is_feas(cone_k)
                return false
            end
        end

        duals_k = solver.dual_views[k]
        # duals_k = copy(solver.dual_views[k]) # TODO prealloc or modify dual_views
        g_k = Cones.grad(cone_k)
        @. duals_k += g_k * solver.mu

        k_nbhd = abs2(norm(duals_k, Inf) / norm(g_k, Inf))
        # k_nbhd = abs2(maximum(abs(dj) / abs(gj) for (dj, gj) in zip(duals_k, g_k))) # TODO try this neighborhood
        if k_nbhd > rhs_nbhd
            return false
        end
    end

    return true
end

# return affine or combined directions, depending on stepper.in_affine_phase
# TODO try to refactor the iterative refinement
function get_directions(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
    rhs = stepper.rhs
    dir = stepper.dir
    dir_temp = stepper.dir_temp
    res = stepper.res
    system_solver = solver.system_solver

    @timeit solver.timer "update_rhs" update_rhs(stepper, solver) # different for affine vs combined phases
    @timeit solver.timer "solve_system" solve_system(system_solver, solver, dir, rhs)

    # use iterative refinement - note apply_LHS is different for affine vs combined phases
    iter_ref_steps = (stepper.in_affine_phase ? 1 : 3) # TODO handle, maybe change dynamically, try fewer for affine phase

    copyto!(dir_temp, dir)
    res = apply_LHS(stepper, solver) # modifies res
    res .-= rhs
    norm_inf = norm(res, Inf)
    norm_2 = norm(res, 2)
    for i in 1:iter_ref_steps
        if norm_inf < 100 * eps(T) # TODO change tolerance dynamically
            break
        end
        @timeit solver.timer "solve_system" solve_system(system_solver, solver, dir, res)
        axpby!(true, dir_temp, -1, dir)
        res = apply_LHS(stepper, solver) # modifies res
        res .-= rhs

        norm_inf_new = norm(res, Inf)
        norm_2_new = norm(res, 2)
        if norm_inf_new > norm_inf || norm_2_new > norm_2
            # residual has not improved
            copyto!(dir, dir_temp)
            break
        end

        # residual has improved, so use the iterative refinement
        solver.verbose && @printf("iter ref round %d norms: inf %9.2e to %9.2e, two %9.2e to %9.2e\n", i, norm_inf, norm_inf_new, norm_2, norm_2_new)
        copyto!(dir_temp, dir)
        norm_inf = norm_inf_new
        norm_2 = norm_2_new
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

            # @show cone_k.dist, cone_k.dual_dist
            # these should be identical for any primal/dual pair, but they are not
            # lambda1 = Cones.scalmat_prod!(similar(cone_k.point), cone_k.dual_point, cone_k)
            # lambda2 = Cones.scalmat_ldiv!(similar(cone_k.point), cone_k.point, cone_k)
            # @show lambda1 ./ lambda2
            # @show lambda1 .- lambda2

            # these updates should be identical
            # @. stepper.s_rhs_k[k] += gamma_mu * tmp2
            # @. stepper.s_rhs_k[k] -= gamma_mu * grad_k

            if Cones.use_scaling(cone_k)
                e1 = similar(cone_k.point)
                Cones.set_initial_point(e1, cone_k)
                lambda = Cones.scalmat_prod!(similar(e1), cone_k.dual_point, cone_k)
                lambda_circ_lambda = Cones.conic_prod!(e1, lambda, lambda, cone_k)
                mehrotra_s = Cones.scalmat_ldiv!(similar(e1), stepper.s_dir_k[k], cone_k)
                mehrotra_z = Cones.scalmat_prod!(similar(e1), stepper.z_dir_k[k], cone_k)
                mehrotra_full = Cones.conic_prod!(similar(e1), mehrotra_s, mehrotra_z, cone_k)
                ds = -lambda_circ_lambda - mehrotra_full + gamma_mu * e1
                ds_by_lambda = Cones.scalvec_ldiv!(similar(e1), ds, cone_k)

                stepper.s_rhs_k[k] .= Cones.scalmat_ldiv!(similar(e1), ds_by_lambda, cone_k)
                #
                # corr = Cones.correction(cone_k, stepper.s_dir_k[k], stepper.z_dir_k[k])
                # @. stepper.s_rhs_k[k] -= corr
            else
                @. stepper.s_rhs_k[k] -= gamma_mu * grad_k
            end
        end

        # kap rhs (with Mehrotra correction)
        rhs[end] += (gamma_mu - stepper.dir[stepper.tau_row] * stepper.dir[stepper.kap_row]) / solver.kap
    end

    return rhs
end

# calculate residual on 6x6 linear system
function apply_LHS(stepper::ScalingStepper{T}, solver::Solver{T}) where {T <: Real}
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
    stepper.res[stepper.kap_row] = tau_dir + solver.tau / solver.kap * kap_dir

    return stepper.res
end

# # TODO experimental for BlockMatrix LHS: if block is a Cone then define mul as hessian product, if block is solver then define mul by mu/tau/tau
# # TODO optimize... maybe need for each cone a 5-arg hess prod
# import LinearAlgebra.mul!
#
# TODO not working because hessian needs to be multiplied by mu
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
#     @. y += alpha * x * solver.tau / solver.kap
#     return y
# end
