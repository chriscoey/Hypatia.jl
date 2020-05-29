
# TODO later move to Cones.jl or elsewhere

use_scaling(cone::Cone) = false

use_correction(cone::Cone) = false

# scal_hess(cone::Cone{T}, mu::T) where {T} = (cone.scal_hess_updated ? cone.scal_hess : update_scal_hess(cone, mu))
scal_hess(cone::Cone{T}, mu::T) where {T} = (cone.scal_hess_updated ? cone.hess : update_scal_hess(cone, mu))

use_update_1_default() = true
# use_update_2_default() = false



outer_prod1(a::AbstractVector{T}, B::AbstractMatrix{T}, alpha::Real) where {T <: LinearAlgebra.BlasReal} = BLAS.syr!('U', alpha, a, B)
outer_prod1(a::AbstractVector{T}, B::AbstractMatrix{T}, alpha::Real) where {T <: Real} = mul!(B, a, a', alpha, true)


# no cholesky updates
function update_scal_hess(
    cone::Cone{T},
    mu::T,
    use_update_1::Bool = use_update_1_default(),
    # use_update_2::Bool = use_update_2_default(),
    ) where {T}
    @assert !cone.scal_hess_updated
    s = cone.point
    z = cone.dual_point

    rtmu = sqrt(mu)
    # irtmu = inv(rtmu)
    # s = rtmu * s

    sz = rtmu * dot(s, z)
    nu = get_nu(cone)

    g = grad(cone)
    # scal_hess = mu * hess(cone)
    scal_hess = hess(cone)

    # TODO tune
    # update_tol = 1e-12
    # update_tol = eps(T)
    # update_tol = 1e-3 * sqrt(eps(T))
    update_tol = 1e5 * eps(T)
    denom_tol = 1e2 * eps(T)

    # first update
    if use_update_1 && norm(z + rtmu * g) > update_tol # TODO should this be more scale-independent? eg normalized, or check dot(Hs, z) / norm(Hs) / norm(z)
        if sz > denom_tol
            # scal_hess += inv(sz) * Symmetric(z * z') - (mu / nu) * Symmetric(g * g')
            # za = inv(sqrt(sz)) * z
            za = z / sqrt(sz)
            # BLAS.syr!('U', one(T), za, scal_hess.data)
            outer_prod1(za, scal_hess.data, one(T))
            # scal_hess += Symmetric(za * za')
            # gb = inv(sqrt(nu)) * g
            # BLAS.syr!('U', -inv(nu), g, scal_hess.data)
            outer_prod1(g, scal_hess.data, -inv(nu))
            # scal_hess -= Symmetric(gb * gb')
            # if norm(scal_hess * rtmu * s - z) > 1e-4
            #     println("large residual after 1st update on norm(scal_hess * s - z): $(norm(scal_hess * s - z))")
            #     # error()
            #     @show norm(scal_hess)
            # end
        # else
        #     println("skipped 1st update (small denom: $sz)")
        end
    # else
        # println("skipped 1st update")
    end

    # @show extrema(eigvals(scal_hess))
    # @show scal_hess

    # second update
    # use_simplifications = true # TODO use?
    # if use_update_2
    #     conj_g = dual_grad(cone, mu)
    #     gsgz = dot(g, conj_g)
    #     cone.dual_grad_inacc && @warn("dual grad inacc in 2nd update")
    #     if !cone.dual_grad_inacc && norm(scal_hess * conj_g - g) > update_tol
    #         mu_cone = sz / nu
    #         # rtmu = sqrt(mu_cone)
    #         # invrtmu = inv(rtmu)
    #         # du_gap = invrtmu * z + rtmu * g
    #         # pr_gap = invrtmu * s + rtmu * conj_g
    #         du_gap = z + mu_cone * g
    #         pr_gap = s + mu_cone * conj_g
    #         # mu_cone * (mu_cone * gsgz - 2nu)
    #         denom_a = (use_simplifications ? abs2(mu_cone) * gsgz - sz : dot(pr_gap, du_gap))
    #         # denom_a = sz + abs2(mu_cone) * gsgz - 2 * nu * mu_cone
    #         # @assert isapprox(denom_a, abs2(mu_cone) * gsgz - sz)
    #
    #         Hgz = hess(cone) * conj_g
    #         if update_one_applied
    #             # struggling with correcness of this
    #             # rho = -conj_g - gsgz / nu * s
    #             # @show norm(s' * mu * hess(cone) * -conj_g - gsgz * mu) # looks BAD
    #             # @show norm(rho - (-conj_g - (s' * mu * hess(cone) * -conj_g) / (s' * mu * hess(cone) * s)  * s)) # looks BAD
    #             # @show norm(scal_hess * -conj_g - (mu * hess(cone) * rho + 1 / mu_cone * z)) # looks BAD
    #             # @show norm(scal_hess * pr_gap - (-mu * mu_cone * hess(cone) * rho)) # looks OK
    #             # @show norm(scal_hess * pr_gap - (mu * mu_cone * (hess(cone) * conj_g - gsgz / nu * g))) # looks OK
    #             H1pg = (use_simplifications ? mu * mu_cone * (Hgz - gsgz / nu * g) : scal_hess * pr_gap)
    #             # @show norm(denom_b - mu * mu_cone^2 * (dot(conj_g, Hgz) - abs2(gsgz) / nu))
    #             denom_b = mu * mu_cone * (dot(conj_g, Hgz) - abs2(gsgz) / nu) * mu_cone
    #         else
    #             # would need to work out alternative simplifications if we want to have this path
    #             H1pg = scal_hess * pr_gap
    #             denom_b = dot(pr_gap, H1pg)
    #         end
    #
    #         if denom_a > denom_tol && denom_b > denom_tol
    #             dga = du_gap / sqrt(denom_a)
    #             scal_hess += Symmetric(dga * dga')
    #             Hpga = H1pg / sqrt(denom_b)
    #             scal_hess -= Symmetric(Hpga * Hpga')
    #             # @show norm(scal_hess * s - z) / (1 + max(norm(s), norm(scal_hess * s)))
    #             # @show norm(scal_hess * -conj_g + g) / (1 + max(norm(g), norm(scal_hess * -conj_g)))
    #         else
    #             @warn("skipped 2nd update (small denoms: $denom_a, $denom_b)")
    #             @show norm(scal_hess * conj_g - g)
    #         end
    #         # @show norm(scal_hess * s - z) / (1 + max(norm(s), norm(scal_hess * s)))
    #         # # @show norm(scal_hess * -conj_g + g)
    #         # @show norm(scal_hess * -conj_g + g) / (1 + max(norm(g), norm(scal_hess * -conj_g)))
    #         # @show norm(scal_hess * pr_gap - du_gap)
    #         # norm(scal_hess * s - z) > 1e-3 || norm(scal_hess * -conj_g + g) > 1e-3  && error()
    #     # else
    #     #     @warn("skipped 2nd update (already satisfied)")
    #     end
    # end

    # TODO OLD:

    # if use_update_1
    #     # Hs = scal_hess * s
    #     # @show norm(Hs + mu * g)
    #     Hs = -mu * g
    #     if norm(Hs - z) > update_tol
    #         # first update
    #         denom_a = dot(s, z)
    #         denom_b = dot(s, Hs)
    #         if denom_a > denom_tol && denom_b > denom_tol
    #             za = z / sqrt(denom_a)
    #             scal_hess += Symmetric(za * za')
    #             Hsb = Hs / sqrt(denom_b)
    #             scal_hess -= Symmetric(Hsb * Hsb')
    #         else
    #             @warn("skipped 1st update (small denoms)")
    #         end
    #         # @show norm(scal_hess * s - z)
    #     # else
    #     #     @warn("skipped 1st update (already satisfied)")
    #     end
    # end

    # if use_update_2
    #     # TODO maybe there are simplifications that can be made here
    #     conj_g = dual_grad(cone, mu)
    #     if !cone.dual_grad_inacc && norm(scal_hess * conj_g - g) > update_tol
    #         # TODO decide whether to use mu_cone = mu or mu_cone = s'z / nu
    #         mu_cone = dot(s, z) / get_nu(cone)
    #         # mu_cone = mu
    #         # rtmu = sqrt(mu_cone)
    #         # invrtmu = inv(rtmu)
    #         # du_gap = invrtmu * z + rtmu * g
    #         # pr_gap = invrtmu * s + rtmu * conj_g
    #         du_gap = z + mu_cone * g
    #         pr_gap = s + mu_cone * conj_g
    #         # second update
    #         denom_a = dot(pr_gap, du_gap)
    #         H1pg = scal_hess * pr_gap # TODO try to mathematically simplify by expanding out scal_hess components
    #         denom_b = dot(pr_gap, H1pg)
    #         # @show mu_cone
    #         # @show du_gap
    #         # @show pr_gap
    #         if denom_a > denom_tol && denom_b > denom_tol
    #             # scal_hess += inv(denom_a) * Symmetric(du_gap * du_gap') - inv(denom_b) * Symmetric(H1pg * H1pg')
    #             dga = du_gap / sqrt(denom_a)
    #             scal_hess += Symmetric(dga * dga')
    #             Hpga = H1pg / sqrt(denom_b)
    #             scal_hess -= Symmetric(Hpga * Hpga')
    #             # dga = inv(sqrt(denom_a)) * du_gap
    #             # scal_hess += Symmetric(dga * dga')
    #             # Hpga = inv(sqrt(denom_b)) * H1pg
    #             # scal_hess -= Symmetric(Hpga * Hpga')
    #             # @show norm(scal_hess * s - z) #/ (1 + max(norm(s), norm(scal_hess * s)))
    #             # @show norm(scal_hess * -conj_g + g) #/ (1 + max(norm(g), norm(scal_hess * -conj_g)))
    #         else
    #             println("skipped 2nd update (small denoms: $denom_a, $denom_b)")
    #             # @show norm(scal_hess * conj_g - g)
    #         end
    #         # @show norm(scal_hess * s - z) / (1 + max(norm(s), norm(scal_hess * s)))
    #         # # @show norm(scal_hess * -conj_g + g)
    #         # @show norm(scal_hess * -conj_g + g) / (1 + max(norm(g), norm(scal_hess * -conj_g)))
    #         # @show norm(scal_hess * pr_gap - du_gap)
    #         # norm(scal_hess * s - z) > 1e-3 || norm(scal_hess * -conj_g + g) > 1e-3  && error()
    #     # else
    #     #     @warn("skipped 2nd update (already satisfied)")
    #     end
    # end


    # copyto!(cone.scal_hess, scal_hess)
    # copyto!(cone.hess, scal_hess)
    cone.scal_hess_updated = true
    # return cone.scal_hess

    return cone.hess
end

# function update_scal_hess(
#     cone::Cone{T},
#     mu::T,
#     use_update_1::Bool = use_update_1_default(),
#     use_update_2::Bool = use_update_2_default(),
#     ) where {T}
#     @assert is_feas(cone)
#     @assert !cone.scal_hess_updated
#     update_tol = 1e-12
#
#     s = cone.point
#     z = cone.dual_point
#     sz = dot(s, z)
#     mu_cone = sz / get_nu(cone)
#     H = hess(cone)
#     scal_hess = Symmetric(mu * H)
#     Hs = scal_hess * s
#     g = grad(cone)
#     conj_g = dual_grad(cone)
#     pr_gap = s + mu_cone * conj_g
#     du_gap = z + mu_cone * g
#     f = cholesky(scal_hess)
#     U = f.U
#     s_tilde = -conj_g
#     rho = s_tilde - (s' * scal_hess * s_tilde) / dot(s, Hs) * s
#
#     if norm(Hs - z) > update_tol && norm(scal_hess * conj_g - g) > update_tol # TODO
#         Us = U * s
#         Urho = U * rho
#         # @assert dot(Us, Urho) <= sqrt(eps(T))
#         r1term = I - (Us * Us') / sum(abs2, Us) - (Urho * Urho') / sum(abs2, Urho)
#         # @show diag(qr(r1term).R) #, rank(r1term)
#         f1_fact = r1term[:, 1] / norm(r1term[:, 1])
#         # @show norm(I - (Us * Us') / sum(abs2, Us) - (Urho * Urho') / sum(abs2, Urho) - y * y')
#         final_col = f.L * f1_fact
#
#         W = vcat(
#             z' / sqrt(sz),
#             dual_gap' / sqrt(dot(primal_gap, dual_gap)),
#             final_col',
#             )
#
#         scal_hess = Symmetric(W' * W)
#         # @show norm(scal_hess * s - z)
#         # @show norm(scal_hess * -conj_g + g)
#     end
#
#
#     copyto!(cone.scal_hess, scal_hess)
#
#     cone.scal_hess_updated = true
#     return cone.scal_hess
# end

# cholesky updates
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
#     # TODO tune
#     update_tol = 1e-12
#     # update_tol = eps(T)
#     # update_tol = sqrt(eps(T))
#     denom_tol = update_tol
#
#     if use_update_1
#         # first update
#         Hs = scal_hess * s
#         if norm(Hs - z) > update_tol
#             denom_a = dot(s, z)
#             denom_b_sqrt = norm(F.U * s)
#             if denom_a > update_tol && abs2(denom_b_sqrt) > update_tol
#                 lowrankupdate!(F, z / sqrt(denom_a))
#                 lowrankdowndate!(F, Hs / denom_b_sqrt)
#             end
#         end
#     end
#
#     if use_update_2
#         # second update
#         g = grad(cone)
#         conj_g = dual_grad(cone)
#         # check gradient of the optimization problem is small
#         # @show norm(ForwardDiff.gradient(barrier(cone), -conj_g) + z)
#         if norm(F.U' * (F.U * conj_g) - g) > update_tol
#             mu_cone = dot(s, z) / get_nu(cone)
#             du_gap = z + mu_cone * g
#             pr_gap = s + mu_cone * conj_g
#
#             denom_a = dot(pr_gap, du_gap)
#             Uprgap = F.U * pr_gap
#             H1pg = F.U' * Uprgap
#             denom_b_sqrt = norm(Uprgap)
#             if denom_a > update_tol && abs2(denom_b_sqrt) > update_tol
#                 lowrankupdate!(F, du_gap / sqrt(denom_a))
#                 lowrankdowndate!(F, H1pg / denom_b_sqrt)
#             end
#         end
#     end
#
#     scal_hess = Symmetric(F.U' * F.U)
#     # @show norm(scal_hess * s - z)
#     # @show norm(scal_hess * conj_g - g)
#     # @show norm(scal_hess * pr_gap - du_gap)
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
    ) where {T}
    mul!(prod, scal_hess(cone, mu), arr)
    return prod
end

function scal_hess_sqrt_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::Cone{T},
    mu::T;
    ) where {T}
    if !cone.hess_fact_updated
        update_hess_fact(cone)
    end
    copyto!(prod, arr)
    sqrt_prod(cone.hess_fact_cache, prod)
    return prod
end


# # TODO make efficient - need an update_scal_hess_prod function and save fields etc
# function scal_hess_prod!(
#     prod::AbstractVecOrMat{T},
#     arr::AbstractVecOrMat{T},
#     cone::Cone{T},
#     mu::T;
#     use_update_1::Bool = use_update_1_default(),
#     use_update_2::Bool = use_update_2_default(),
#     ) where {T}
#     s = cone.point
#     z = cone.dual_point
#
#     hess_prod!(prod, arr, cone)
#     @. prod *= mu
#
#     # TODO tune
#     update_tol = 1e-12
#     # update_tol = eps(T)
#     # update_tol = sqrt(eps(T))
#     denom_tol = update_tol
#
#     if use_update_1
#         Hs = similar(s)
#         hess_prod!(Hs, s, cone)
#         @. Hs *= mu
#         if norm(Hs - z) > update_tol
#             # first update
#             denom_a = dot(s, z)
#             denom_b = dot(s, Hs)
#             if denom_a > denom_tol && denom_b > denom_tol
#                 for j in 1:size(arr, 2)
#                     @views arrj = arr[:, j]
#                     scale_a = dot(z, arrj) / denom_a
#                     scale_b = dot(Hs, arrj) / denom_b
#                     @. prod[:, j] += scale_a * z
#                     @. prod[:, j] -= scale_b * Hs
#                 end
#             end
#         end
#     end
#
#     if use_update_2
#         g = grad(cone)
#         conj_g = dual_grad(cone)
#         H1cg = similar(g)
#         scal_hess_prod!(H1cg, conj_g, cone, mu, use_update_1 = true, use_update_2 = false)
#         if norm(H1cg - g) > update_tol
#             mu_cone = dot(s, z) / get_nu(cone)
#             du_gap = z + mu_cone * g
#             pr_gap = s + mu_cone * conj_g
#             # second update
#             denom_a = dot(pr_gap, du_gap)
#             H1pg = similar(s)
#             scal_hess_prod!(H1pg, pr_gap, cone, mu, use_update_1 = true, use_update_2 = false)
#             denom_b = dot(pr_gap, H1pg)
#             if denom_a > denom_tol && denom_b > denom_tol
#                 for j in 1:size(arr, 2)
#                     @views arrj = arr[:, j]
#                     scale_a = dot(du_gap, arrj) / denom_a
#                     scale_b = dot(H1pg, arrj) / denom_b
#                     @. prod[:, j] += scale_a * du_gap
#                     @. prod[:, j] -= scale_b * H1pg
#                 end
#             end
#         end
#     end
#
#     return prod
# end

# # correction fallback (TODO remove later)
# import ForwardDiff
#
# function correction(cone::Cone{T}, primal_dir::AbstractVector{T}, dual_dir::AbstractVector{T}) where {T}
#     dim = cone.dim
#     point = cone.point
#     FD_3deriv = ForwardDiff.jacobian(x -> ForwardDiff.hessian(barrier(cone), x), point)
#     # check log-homog property that F'''(point)[point] = -2F''(point)
#     @assert reshape(FD_3deriv * cone.point, dim, dim) ≈ -2 * ForwardDiff.hessian(barrier(cone), point)
#     Hinv_z = inv_hess_prod!(similar(dual_dir), dual_dir, cone)
#     FD_corr = reshape(FD_3deriv * primal_dir, dim, dim) * Hinv_z / -2
#     return FD_corr
# end
