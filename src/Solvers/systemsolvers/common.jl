#=
linear system solver subroutines

6x6 nonsymmetric system in (x, y, z, tau, s, kap):
A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
-G*x + h*tau - s = zrhs
-c'*x - b'*y - h'*z - kap = taurhs
(pr bar) z_k + mu*H_k*s_k = srhs_k
(du bar) mu*H_k*z_k + s_k = srhs_k
mu/(taubar^2)*tau + kap = kaprhs
=#

# calculate direction given rhs, and apply iterative refinement
function get_directions(
    stepper::Stepper{T},
    solver::Solver{T},
    use_nt::Bool;
    iter_ref_steps::Int = 0,
    ) where {T <: Real}
    rhs = stepper.rhs
    dir = stepper.dir
    dir_temp = stepper.dir_temp
    res = stepper.res
    system_solver = solver.system_solver

    tau = solver.point.tau[1]
    tau_scal = (use_nt ? solver.point.kap[1] : solver.mu / tau) / tau

    solve_system(system_solver, solver, dir, rhs, tau_scal)

    # use iterative refinement
    copyto!(dir_temp, dir.vec)
    apply_lhs(stepper, solver, tau_scal) # modifies res
    res.vec .-= rhs.vec
    norm_inf = norm(res.vec, Inf)
    norm_2 = norm(res.vec, 2)

    for i in 1:iter_ref_steps
        # @show norm_inf
        if norm_inf < 100 * eps(T) # TODO change tolerance dynamically
            break
        end
        solve_system(system_solver, solver, dir, res, tau_scal)
        axpby!(true, dir_temp, -1, dir.vec)
        res = apply_lhs(stepper, solver, tau_scal) # modifies res
        res.vec .-= rhs.vec

        norm_inf_new = norm(res.vec, Inf)
        norm_2_new = norm(res.vec, 2)
        if norm_inf_new > norm_inf || norm_2_new > norm_2
            # residual has not improved
            copyto!(dir.vec, dir_temp)
            break
        end

        # residual has improved, so use the iterative refinement
        copyto!(dir_temp, dir.vec)
        norm_inf = norm_inf_new
        norm_2 = norm_2_new
    end

    # if norm_inf > solver.mu
    #     @warn("bad residual on direction: $norm_inf")
    #     for (k, cone_k) in enumerate(solver.model.cones)
    #         # if norm(Cones.hess(cone_k) * Cones.inv_hess(cone_k) - I, Inf) > 1e-10
    #         #     @show k
    #         #     # @show cone_k.point
    #         #     @show cone_k.z
    #         #     @show cone_k.lwv
    #         #     @show cone_k.grad
    #         #     # @show cone_k.point[cone_k.v_idxs]
    #         #     # @show cone_k.point[cone_k.w_idxs]
    #         #     @show Cones.hess(cone_k) * cone_k.point + Cones.grad(cone_k)
    #         @show dot(Cones.grad(cone_k), cone_k.point) + Cones.get_nu(cone_k)
    #         # @show norm(Cones.hess(cone_k) * cone_k.point + Cones.grad(cone_k), Inf)
    #         # @show norm(Cones.inv_hess(cone_k) * Cones.grad(cone_k) + cone_k.point, Inf)
    #         temp = similar(cone_k.point)
    #         @show cone_k.grad_updated
    #         @show norm(Cones.hess_prod!(temp, cone_k.point, cone_k) + cone_k.grad, Inf)
    #         @show norm(Cones.inv_hess_prod!(temp, cone_k.grad, cone_k) + cone_k.point, Inf)
    #         @show norm(Cones.hess(cone_k) * Cones.inv_hess(cone_k) - I)
    #         # end
    #         if norm(Cones.hess(cone_k) * Cones.inv_hess(cone_k) - I, Inf) > 1e-10
    #             @show Cones.hess(cone_k) * Cones.inv_hess(cone_k)
    #         end
    #     end
    #     @show norm(res.x, Inf)
    #     # @show norm(res.y, Inf)
    #     # @show norm(res.z, Inf)
    #     # @show norm(res.tau[1], Inf)
    #     # @show norm(res.s, Inf)
    #     # @show norm(res.kap[1], Inf)
    # end
    # println()
    @assert !isnan(norm_inf) # TODO error

    return dir
end

# calculate residual on 6x6 linear system
function apply_lhs(
    stepper::Stepper{T},
    solver::Solver{T},
    tau_scal::T,
    ) where {T <: Real}
    model = solver.model
    dir = stepper.dir
    res = stepper.res
    tau_dir = dir.tau[1]
    kap_dir = dir.kap[1]

    # A'*y + G'*z + c*tau
    copyto!(res.x, model.c)
    mul!(res.x, model.G', dir.z, true, tau_dir)
    # -G*x + h*tau - s
    @. res.z = model.h * tau_dir - dir.s
    mul!(res.z, model.G, dir.x, -1, true)
    # -c'*x - b'*y - h'*z - kap
    res.tau[1] = -dot(model.c, dir.x) - dot(model.h, dir.z) - kap_dir
    # if p = 0, ignore A, b, y
    if !iszero(model.p)
        # A'*y + G'*z + c*tau
        mul!(res.x, model.A', dir.y, true, true)
        # -A*x + b*tau
        copyto!(res.y, model.b)
        mul!(res.y, model.A, dir.x, -1, tau_dir)
        # -c'*x - b'*y - h'*z - kap
        res.tau[1] -= dot(model.b, dir.y)
    end

    # s
    for (k, cone_k) in enumerate(model.cones)
        # (du bar) mu*H_k*z_k + s_k
        # (pr bar) z_k + mu*H_k*s_k
        s_res_k = res.s_views[k]
        Cones.hess_prod!(s_res_k, dir.primal_views[k], cone_k)
        @. s_res_k += dir.dual_views[k]
    end

    res.kap[1] = tau_scal * tau_dir + kap_dir

    return stepper.res
end

include("naive.jl")
include("naiveelim.jl")
include("symindef.jl")
include("qrchol.jl")

function solve_inner_system(
    system_solver::Union{NaiveElimSparseSystemSolver, SymIndefSparseSystemSolver},
    sol::Point,
    rhs::Point,
    )
    inv_prod(system_solver.fact_cache, sol.vec, system_solver.lhs_sub, rhs.vec)
    return sol
end

function solve_inner_system(
    system_solver::Union{NaiveElimDenseSystemSolver, SymIndefDenseSystemSolver},
    sol::Point,
    rhs::Point,
    )
    copyto!(sol.vec, rhs.vec)
    inv_prod(system_solver.fact_cache, sol.vec)
    return sol
end

# reduce to 4x4 subsystem
function solve_system(
    system_solver::Union{NaiveElimSystemSolver{T}, SymIndefSystemSolver{T}, QRCholSystemSolver{T}},
    solver::Solver{T},
    sol::Point{T},
    rhs::Point{T},
    tau_scal::T,
    ) where {T <: Real}
    model = solver.model

    solve_subsystem4(system_solver, solver, sol, rhs, tau_scal)
    tau = sol.tau[1]

    # lift to get s and kap
    # s = -G*x + h*tau - zrhs
    @. sol.s = model.h * tau - rhs.z
    mul!(sol.s, model.G, sol.x, -1, true)

    # kap = -kapbar/taubar*tau + kaprhs
    sol.kap[1] = -tau_scal * tau + rhs.kap[1]

    return sol
end

# reduce to 3x3 subsystem
function solve_subsystem4(
    system_solver::Union{SymIndefSystemSolver{T}, QRCholSystemSolver{T}},
    solver::Solver{T},
    sol::Point{T},
    rhs::Point{T},
    tau_scal::T,
    ) where {T <: Real}
    model = solver.model
    rhs_sub = system_solver.rhs_sub
    sol_sub = system_solver.sol_sub

    @. rhs_sub.x = rhs.x
    @. rhs_sub.y = -rhs.y
    setup_rhs3(system_solver, model, rhs, sol, rhs_sub)

    solve_subsystem3(system_solver, solver, sol_sub, rhs_sub)

    # lift to get tau
    sol_const = system_solver.sol_const
    tau_num = rhs.tau[1] + rhs.kap[1] + dot_obj(model, sol_sub)
    tau_denom = tau_scal - dot_obj(model, sol_const)
    sol_tau = tau_num / tau_denom

    dim3 = length(sol_sub.vec)
    @. @views sol.vec[1:dim3] = sol_sub.vec + sol_tau * sol_const.vec
    sol.tau[1] = sol_tau

    return sol
end

function setup_point_sub(system_solver::Union{QRCholSystemSolver{T}, SymIndefSystemSolver{T}}, model::Models.Model{T}) where {T <: Real}
    (n, p, q) = (model.n, model.p, model.q)
    dim_sub = n + p + q
    z_start = model.n + model.p
    sol_sub = system_solver.sol_sub = Point{T}()
    rhs_sub = system_solver.rhs_sub = Point{T}()
    rhs_const = system_solver.rhs_const = Point{T}()
    sol_const = system_solver.sol_const = Point{T}()
    for point_sub in (sol_sub, rhs_sub, rhs_const, sol_const)
        point_sub.vec = zeros(T, dim_sub)
        @views point_sub.x = point_sub.vec[1:n]
        @views point_sub.y = point_sub.vec[n .+ (1:p)]
        @views point_sub.z = point_sub.vec[n + p .+ (1:q)]
        point_sub.z_views = [view(point_sub.z, idxs) for idxs in model.cone_idxs]
    end
    @. rhs_const.x = -model.c
    @. rhs_const.y = model.b
    @. rhs_const.z = model.h
    return nothing
end

dot_obj(model::Models.Model, point::Point) = dot(model.c, point.x) + dot(model.b, point.y) + dot(model.h, point.z)
