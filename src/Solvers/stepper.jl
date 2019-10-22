#=
Copyright 2019, Chris Coey and contributors

interior point stepping routines for algorithms based on homogeneous self dual embedding
=#

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

# function check_nbhd(
#     mu_temp::T,
#     taukap_temp::T,
#     nbhd::T,
#     solver::Solver{T},
#     ) where {T <: Real}
#     cones = solver.model.cones
#     sqrtmu = sqrt(mu_temp)
#
#     rhs_nbhd = mu_temp * abs2(nbhd)
#     lhs_nbhd = abs2(taukap_temp / sqrtmu - sqrtmu)
#     if lhs_nbhd >= rhs_nbhd
#         return false
#     end
#
#     Cones.load_point.(cones, solver.primal_views, sqrtmu)
#     Cones.load_dual_point.(cones, solver.dual_views, sqrtmu) # TODO needed?
#
#     # accept primal iterate if it is inside the cone and neighborhood
#     # first check inside cone for whichever cones were violated last line search iteration
#     for (k, cone_k) in enumerate(cones)
#         if solver.cones_infeas[k]
#             Cones.reset_data(cone_k)
#             if Cones.is_feas(cone_k)
#                 solver.cones_infeas[k] = false
#                 solver.cones_loaded[k] = true
#             else
#                 return false
#             end
#         else
#             solver.cones_loaded[k] = false
#         end
#     end
#
#     for (k, cone_k) in enumerate(cones)
#         if !solver.cones_loaded[k]
#             Cones.reset_data(cone_k)
#             if !Cones.is_feas(cone_k)
#                 return false
#             end
#         end
#
#         # modifies dual_views
#         duals_k = solver.dual_views[k]
#         g_k = Cones.grad(cone_k)
#         @. duals_k += g_k * sqrtmu
#
#         if solver.use_infty_nbhd
#             k_nbhd = abs2(norm(duals_k, Inf) / norm(g_k, Inf))
#             # k_nbhd = abs2(maximum(abs(dj) / abs(gj) for (dj, gj) in zip(duals_k, g_k))) # TODO try this neighborhood
#             lhs_nbhd = max(lhs_nbhd, k_nbhd)
#         else
#             nbhd_temp_k = solver.nbhd_temp[k]
#             Cones.inv_hess_prod!(nbhd_temp_k, duals_k, cone_k)
#             k_nbhd = dot(duals_k, nbhd_temp_k)
#             if k_nbhd <= -cbrt(eps(T))
#                 @warn("numerical failure: cone neighborhood is $k_nbhd")
#                 return false
#             elseif k_nbhd > zero(T)
#                 lhs_nbhd += k_nbhd
#             end
#         end
#
#         if lhs_nbhd > rhs_nbhd
#             return false
#         end
#     end
#
#     return true
# end

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
