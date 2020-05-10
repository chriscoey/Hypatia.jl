
# TODO later move to Cones.jl or elsewhere

use_scaling(cone::Cone) = false

use_correction(cone::Cone) = false

scal_hess(cone::Cone{T}, mu::T) where {T} = (cone.scal_hess_updated ? cone.scal_hess : update_scal_hess(cone, mu))

use_update_1_default() = true
use_update_2_default() = true


# updates cone.hess in place, before factorization
# TODO can't use that cone.hess for the skajaa ye neighborhood
function update_scal_hess(
    cone::Cone{T},
    mu::T,
    use_update_1::Bool = use_update_1_default(),
    use_update_2::Bool = use_update_2_default(),
    ) where {T}
    @assert is_feas(cone)
    @assert !cone.scal_hess_updated
    s = cone.point
    z = cone.dual_point

    scal_hess = mu * hess(cone)

    if use_update_1_default()
        # first update
        denom_a = dot(s, z)
        Hs = scal_hess * s
        denom_b = dot(s, Hs)
        if denom_a > 0
            za = z / sqrt(denom_a)
            scal_hess += Symmetric(za * za')
        end
        if denom_b > 0
            Hsb = Hs / sqrt(denom_b)
            scal_hess -= Symmetric(Hsb * Hsb')
        end
        # @show norm(scal_hess * s - z)
    end

    if use_update_2_default()
        # second update
        g = grad(cone)
        conj_g = dual_grad(cone)
        # check gradient of the optimization problem is small
        # @show norm(ForwardDiff.gradient(cone.barrier, -conj_g) + z)
        mu_cone = dot(s, z) / get_nu(cone)
        # @show mu_cone
        dual_gap = z + mu_cone * g
        primal_gap = s + mu_cone * conj_g
        denom_a = dot(primal_gap, dual_gap)
        H1prgap = scal_hess * primal_gap
        denom_b = dot(primal_gap, H1prgap)
        if denom_a > 0
            dga = dual_gap / sqrt(denom_a)
            scal_hess += Symmetric(dga * dga')
        end
        if denom_b > 0
            Hpga = H1prgap / sqrt(denom_b)
            scal_hess -= Symmetric(Hpga * Hpga')
        end
        # @show norm(scal_hess * s - z)
        # @show norm(scal_hess * -conj_g + g)
        # @show norm(scal_hess * -conj_g + g) / (1 + max(norm(g), norm(scal_hess * -conj_g)))
        # @show norm(scal_hess * primal_gap - dual_gap)
        # norm(scal_hess * s - z) > 1e-3 || norm(scal_hess * -conj_g + g) > 1e-3  && error()
    end

    copyto!(cone.scal_hess, scal_hess)

    cone.scal_hess_updated = true
    return cone.scal_hess
end

#
# function update_scal_hess(
#     cone::Cone{T},
#     mu::T,
#     use_update_1::Bool = use_update_1_default(),
#     use_update_2::Bool = use_update_2_default(),
#     ) where {T}
#     @assert is_feas(cone)
#     @assert !cone.scal_hess_updated
#     s = cone.point
#     z = cone.dual_point
#
#     scal_hess = mu * hess(cone)
#     F = cholesky(scal_hess)
#     # @show mu
#
#     if use_update_1_default()
#         # first update
#         denom_a = dot(s, z)
#         muHs = scal_hess * s
#         denom_b = dot(s, muHs)
#
#         if denom_a > 0
#             scal_hess += Symmetric(z * z') / denom_a
#         end
#         if denom_b > 0
#             # LinearAlgebra.copytri!(scal_hess.data, 'L')
#             # mul!(scal_hess.data, muHs, muHs', -inv(denom_b), 1)
#             scal_hess -= Symmetric(muHs * muHs') / denom_b
#         end
#         # @show norm(scal_hess * s - z)
#     end
#
#     if use_update_2_default()
#         # second update
#         g = grad(cone)
#         conj_g = dual_grad(cone)
#         # check gradient of the optimization problem is small
#         # @show norm(ForwardDiff.gradient(cone.barrier, -conj_g) + z)
#
#         mu_cone = dot(s, z) / get_nu(cone)
#         # @show mu_cone
#         dual_gap = z + mu_cone * g
#         # @show log(-z[3] / z[1]) * z[1] + z[1] - z[2]
#         # @show g
#         # @show dual_gap
#         primal_gap = s + mu_cone * conj_g
#         # @show s[2] * log(s[3] / s[2]) - s[1]
#
#         denom_a = dot(primal_gap, dual_gap)
#         H1prgap = scal_hess * primal_gap
#         denom_b = dot(primal_gap, H1prgap)
#
#         if denom_a > 0
#             scal_hess += Symmetric(dual_gap * dual_gap') / denom_a
#         else
#             # println("DENOM A BAD")
#             # @show denom_a
#         end
#         if denom_b > 0
#             scal_hess -= Symmetric(H1prgap * H1prgap') / denom_b
#         else
#             # println("DENOM B BAD")
#             # @show denom_b
#         end
#         # @show norm(scal_hess * s - z)
#         # @show norm(scal_hess * -conj_g + g)
#         # @show norm(scal_hess * -conj_g + g) / (1 + max(norm(g), norm(scal_hess * -conj_g)))
#         # @show norm(scal_hess * primal_gap - dual_gap)
#         # norm(scal_hess * s - z) > 1e-3 || norm(scal_hess * -conj_g + g) > 1e-3  && error()
#     end
#
#     copyto!(cone.scal_hess, scal_hess)
#
#     cone.scal_hess_updated = true
#     return cone.scal_hess
# end


# function update_scal_hess(
#     cone::Cone{T},
#     mu::T;
#     use_update_1::Bool = use_update_1_default(),
#     use_update_2::Bool = use_update_2_default(),
#     ) where {T}
#     @assert is_feas(cone)
#     @assert !cone.scal_hess_updated
#     s = cone.point
#     z = cone.dual_point
#
#     scal_hess = mu * hess(cone)
#     F = cholesky(Symmetric(Matrix(scal_hess), :U), check = false) # Hess might not be a dense matrix
#     if !issuccess(F)
#         error("cholesky did not succeed in update_scal_hess")
#         flush(stdout)
#     end
#
#     if use_update_1
#         # first update
#         denom_a = dot(s, z)
#         muHs = scal_hess * s
#         denom_b_sqrt = norm(F.U * s)
#
#         if denom_a > 0
#             lowrankupdate!(F, z / sqrt(denom_a))
#         end
#         if denom_b_sqrt > 0
#             lowrankdowndate!(F, muHs / denom_b_sqrt)
#         end
#
#         # @show norm(F.U' * (F.U * s) - z)
#     end
#
#     if use_update_2
#         # second update
#         g = grad(cone)
#         conj_g = dual_grad(cone)
#         # check gradient of the optimization problem is small
#         # @show norm(ForwardDiff.gradient(barrier(cone), -conj_g) + z)
#
#         mu_cone = dot(s, z) / get_nu(cone)
#         dual_gap = z + mu_cone * g
#         primal_gap = s + mu_cone * conj_g
#
#         denom_a = dot(primal_gap, dual_gap)
#         Uprgap = F.U * primal_gap
#         H1prgap = F.U' * Uprgap
#         denom_b_sqrt = norm(Uprgap)
#         if denom_a > 0
#             lowrankupdate!(F, dual_gap / sqrt(denom_a))
#         end
#         if denom_b_sqrt > 0
#             lowrankdowndate!(F, H1prgap / denom_b_sqrt)
#         end
#     end
#
#     scal_hess = Symmetric(F.U' * F.U)
#     # @show norm(scal_hess * s - z)
#     # @show norm(scal_hess * -conj_g + g)
#     # @show norm(scal_hess * primal_gap - dual_gap)
#     # (norm(scal_hess * s - z) > 1e-3 || norm(scal_hess * -conj_g + g) > 1e-3) && error()
#
#     copyto!(cone.scal_hess, scal_hess)
#
#     cone.scal_hess_updated = true
#     return cone.scal_hess
# end


function scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::Cone{T},
    mu::T;
    use_update_1::Bool = use_update_1_default(),
    use_update_2::Bool = use_update_2_default(),
    ) where {T}
    mul!(prod, scal_hess(cone, mu), arr)
    return prod
end

# function scal_hess_prod!(
#     prod::AbstractVecOrMat{T},
#     arr::AbstractVecOrMat{T},
#     cone::Cone{T},
#     mu::T;
#     use_update_1::Bool = use_update_1_default(),
#     use_update_2::Bool = use_update_2_default(),
#     ) where {T}
#     @assert is_feas(cone)
#     s = cone.point
#     z = cone.dual_point
#
#     hess_prod!(prod, arr, cone)
#     @. prod *= mu
#
#     if use_update_1
#         muHs = similar(s)
#         hess_prod!(muHs, s, cone)
#         @. muHs *= mu
#         denom_a = dot(s, z)
#         denom_b = dot(s, muHs)
#         if denom_a > 0 && denom_b > 0
#             for j in 1:size(arr, 2)
#                 @views arrj = arr[:, j]
#                 scale_a = dot(z, arrj) / denom_a
#                 scale_b = dot(muHs, arrj) / denom_b
#                 @. prod[:, j] += scale_a * z
#                 @. prod[:, j] -= scale_b * muHs
#             end
#         end
#     end
#
#     if use_update_2
#         g = grad(cone)
#         conj_g = dual_grad(cone)
#         mu_cone = dot(s, z) / get_nu(cone)
#         primal_gap = s + mu_cone * conj_g
#         dual_gap = z + mu_cone * g
#         # TODO do this in a better way
#         H1prgap = similar(s)
#         scal_hess_prod!(H1prgap, primal_gap, cone, mu, use_update_1 = true, use_update_2 = false)
#         # @show isapprox(H1prgap, update_scal_hess(cone, mu, use_update_1 = true, use_update_2 = false) * primal_gap)
#         denom_a = dot(primal_gap, dual_gap)
#         denom_b = dot(primal_gap, H1prgap)
#         if denom_a > 0 && denom_b > 0
#             for j in 1:size(arr, 2)
#                 @views arrj = arr[:, j]
#                 scale_a = dot(dual_gap, arrj) / denom_a
#                 scale_b = dot(H1prgap, arrj) / denom_b
#                 @. prod[:, j] += scale_a * dual_gap
#                 @. prod[:, j] -= scale_b * H1prgap
#             end
#         end
#     end
#
#     return prod
# end

# correction fallback (TODO remove later)
import ForwardDiff

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
