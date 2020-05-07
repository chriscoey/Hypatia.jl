
# TODO later move to Cones.jl or elsewhere

import Optim
import ForwardDiff
include("newton.jl")

use_scaling(cone::Cone) = false

use_correction(cone::Cone) = false

scal_hess(cone::Cone{T}, mu::T, z::AbstractVector{T}) where {T} = (cone.scal_hess_updated ? cone.scal_hess : update_scal_hess(cone, mu, z))

function update_scal_hess(
    cone::Cone{T},
    mu::T,
    z::AbstractVector{T}; # dual point
    use_update_1::Bool = false, # easy update
    use_update_2::Bool = false, # hard update
    ) where {T}
    @assert is_feas(cone)
    @assert !cone.scal_hess_updated
    s = cone.point
    # z = cone.dual_point

    scal_hess = mu * hess(cone)
    F = cholesky(scal_hess)
    println("##########################################")
    # @show mu

    if use_update_1
        # first update
        denom_a = dot(s, z)
        muHs = scal_hess * s
        denom_b = dot(s, muHs)

        if denom_a > 0
            scal_hess += Symmetric(z * z') / denom_a
        end
        if denom_b > 0
            # LinearAlgebra.copytri!(scal_hess.data, 'L')
            # mul!(scal_hess.data, muHs, muHs', -inv(denom_b), 1)
            scal_hess -= Symmetric(muHs * muHs') / denom_b
        end
        @show norm(scal_hess * s - z)
    end

    if use_update_2
        # second update
        g = grad(cone)
        conj_g = conjugate_gradient2(barrier(cone), s, z)
        # check gradient of the optimization problem is small
        # @show norm(ForwardDiff.gradient(barrier(cone), -conj_g) + z)

        mu_cone = dot(s, z) / get_nu(cone)
        # @show mu_cone
        dual_gap = z + mu_cone * g
        # @show log(-z[3] / z[1]) * z[1] + z[1] - z[2]
        # @show g
        # @show dual_gap
        primal_gap = s + mu_cone * conj_g
        # @show s[2] * log(s[3] / s[2]) - s[1]

        denom_a = dot(primal_gap, dual_gap)
        H1prgap = scal_hess * primal_gap
        denom_b = dot(primal_gap, H1prgap)

        if denom_a > 0
            scal_hess += Symmetric(dual_gap * dual_gap') / denom_a
        else
            println("DENOM A BAD")
            @show denom_a
        end
        if denom_b > 0
            scal_hess -= Symmetric(H1prgap * H1prgap') / denom_b
        else
            println("DENOM B BAD")
            @show denom_b
        end
        @show norm(scal_hess * s - z)
        @show norm(scal_hess * -conj_g + g)
        @show norm(scal_hess * -conj_g + g) / (1 + max(norm(g), norm(scal_hess * -conj_g)))
        @show norm(scal_hess * primal_gap - dual_gap)
        # norm(scal_hess * s - z) > 1e-3 || norm(scal_hess * -conj_g + g) > 1e-3  && error()
    end

    copyto!(cone.scal_hess, scal_hess)

    cone.scal_hess_updated = true
    return cone.scal_hess
end


# function update_scal_hess(
#     cone::Cone{T},
#     mu::T,
#     z::AbstractVector{T}; # dual point
#     use_update_1::Bool = true, # easy update
#     use_update_2::Bool = true, # hard update
#     ) where {T}
#     @assert is_feas(cone)
#     @assert !cone.scal_hess_updated
#     s = cone.point
#     # z = cone.dual_point
#
#     scal_hess = mu * hess(cone)
#     F = cholesky(scal_hess)
#     println("##########################################")
#
#     if use_update_1
#         # first update
#         denom_a = dot(s, z)
#         muHs = scal_hess * s
#         denom_b_sqrt = sqrt(sum(abs2, F.U * s))
#
#         if denom_a > 0
#             lowrankupdate!(F, z / sqrt(denom_a))
#         end
#         if denom_b_sqrt > 0
#             lowrankdowndate!(F, muHs / denom_b_sqrt)
#         end
#         scal_hess_1 = Symmetric(F.L * F.U)
#         @show norm(scal_hess_1 * s - z)
#     end
#
#     if use_update_2
#         # second update
#         g = grad(cone)
#         conj_g = conjugate_gradient2(barrier(cone), s, z)
#         # check gradient of the optimization problem is small
#         @show norm(ForwardDiff.gradient(barrier(cone), -conj_g) + z)
#
#         mu_cone = dot(s, z) / get_nu(cone)
#         dual_gap = z + mu_cone * g
#         primal_gap = s + mu_cone * conj_g
#
#         denom_a = dot(primal_gap, dual_gap)
#         H1prgap = scal_hess_1 * primal_gap
#         denom_b_sqrt = sqrt(sum(abs2, F.U * primal_gap))
#         if denom_a > 0
#             lowrankupdate!(F, dual_gap / sqrt(denom_a))
#         end
#         if denom_b_sqrt > 0
#             lowrankdowndate!(F, H1prgap / denom_b_sqrt)
#         end
#
#         scal_hess = Symmetric(F.L * F.U)
#         @show norm(scal_hess * s - z)
#         @show norm(scal_hess * -conj_g + g)
#         @show norm(scal_hess * primal_gap - dual_gap)
#         # norm(scal_hess * s - z) > 1e-3 || norm(scal_hess * -conj_g + g) > 1e-3  && error()
#     end
#
#     copyto!(cone.scal_hess, scal_hess)
#
#     cone.scal_hess_updated = true
#     return cone.scal_hess
# end

# TODO use domain constraints properly
# TODO use central point as starting point?
# TODO use constrained method from https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/ipnewton_basics/#constrained-optimization-with-ipnewton
function conjugate_gradient1(barrier::Function, s::AbstractVector{T}, z::AbstractVector{T}) where {T}
    modified_legendre(x) = ((x[2] <= 0 || x[3] <= 0 || x[2] * log(x[3] / x[2]) - x[1] <= 0) ? 1e16 : dot(z, x) + barrier(x)) # TODO bad feas check - need constraints
    res = Optim.optimize(modified_legendre, s, Optim.Newton())
    # res = Optim.optimize(modified_legendre, [-0.827838399, 0.805102005, 1.290927713], Optim.Newton())
    minimizer = Optim.minimizer(res)
    @assert !any(isnan, minimizer)
    return -minimizer
end

function conjugate_gradient2(barrier::Function, s::AbstractVector{T}, z::AbstractVector{T}) where {T}
    nc = NewtonCache{T}(barrier, z)
    nc.x = copy(s)
    # nc.x = T[-0.827838399, 0.805102005, 1.290927713]
    # damped_newton_method(nc)
    switched_newton_method(nc)
    return -nc.x
end

# for solving optimization problem only in BF
function conjugate_gradient3(barrier::Function, s::AbstractVector{T}, z::AbstractVector{T}) where {T}
    nc = NewtonCache{BigFloat}(barrier, BigFloat.(z))
    @show nc
    nc.x = BigFloat[-0.827838399, 0.805102005, 1.290927713]
    # damped_newton_method(nc)
    switched_newton_method(nc)
    return -Float64.(nc.x)
end

# correction
function correction(cone::Cone{T}, primal_dir::AbstractVector{T}, dual_dir::AbstractVector{T}) where {T}
    dim = cone.dim
    point = cone.point
    FD_3deriv = ForwardDiff.jacobian(x -> ForwardDiff.hessian(barrier(cone), x), point)
    # check log-homog property that F'''(point)[point] = -2F''(point)
    @assert reshape(FD_3deriv * cone.point, dim, dim) ≈ -2 * ForwardDiff.hessian(barrier(cone), point)
    Hinv_z = inv_hess_prod!(similar(dual_dir), dual_dir, cone)
    FD_corr = reshape(FD_3deriv * primal_dir, dim, dim) * Hinv_z / -2
    return FD_corr
end
