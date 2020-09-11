#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

=#
using ForwardDiff

mutable struct WSOSInterpEpiNormInf{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    point::AbstractVector{T}
    dual_point::AbstractVector{T}
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache
    correction::Vector{T}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    barrier::Function

    mats::Vector{Vector{Matrix{T}}}
    matfact::Vector{Vector}
    Λi_Λ::Vector{Vector{Matrix{T}}}
    Λ11::Vector{Matrix{T}}
    tmpΛ11::Vector{Matrix{T}}
    tmpLL::Vector{Matrix{T}}
    tmpLU::Vector{Matrix{T}}
    tmpLU2::Vector{Matrix{T}}
    tmpUU_vec::Vector{Matrix{T}} # reused in update_hess
    tmpUU::Matrix{T}
    PΛiPs1::Vector{Vector{Matrix{T}}} # for each (2, 2)-block pertaining to (lambda_1, lambda_i), P * inv(Λ)[1, 1] * Ps = P * inv(Λ)i[2, 2] * Ps
    PΛiPs2::Vector{Vector{Matrix{T}}} # for each (2, 2)-block pertaining to (lambda_1, lambda_i), P * inv(Λ)[2, 1] * Ps = P * inv(Λ)[1, 2]' * Ps
    lambdafact::Vector
    point_views

    function WSOSInterpEpiNormInf{T}(
        R::Int,
        U::Int,
        Ps::Vector{Matrix{T}};
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        for Pk in Ps
            @assert size(Pk, 1) == U
        end
        cone = new{T}()
        cone.use_dual_barrier = !use_dual # using dual barrier
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.dim = U * R
        cone.R = R
        cone.U = U
        cone.Ps = Ps
        cone.hess_fact_cache = hess_fact_cache

        # soc-based
        # function barrier(point)
        #      bar = zero(eltype(point))
        #      for P in cone.Ps
        #          lambda_1 = Symmetric(P' * Diagonal(point[1:U]) * P)
        #          fact_1 = cholesky(lambda_1)
        #          for i in 2:R
        #              lambda_i = Symmetric(P' * Diagonal(point[block_idxs(U, i)]) * P)
        #              LL = fact_1.L \ lambda_i
        #              bar -= logdet(lambda_1 - LL' * LL)
        #              # bar -= logdet(lambda_1 - lambda_i * (fact_1 \ lambda_i))
        #          end
        #          bar -= logdet(fact_1)
        #      end
        #      return bar
        # end

        # orthant-based
        function barrier(point)
             bar = zero(eltype(point))
             for P in cone.Ps
                 lambda_1 = Hermitian(P' * Diagonal(point[1:U]) * P)
                 for i in 2:R
                     lambda_i = Hermitian(P' * Diagonal(point[block_idxs(U, i)]) * P)
                     bar -= logdet(lambda_1 - lambda_i) + logdet(lambda_1 + lambda_i)
                 end
                 bar += logdet(cholesky(lambda_1)) * (R - 2)
             end
             return bar
        end

        cone.barrier = barrier

        return cone
    end
end

function setup_data(cone::WSOSInterpEpiNormInf{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    U = cone.U
    R = cone.R
    Ps = cone.Ps
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = similar(cone.point)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.correction = zeros(T, dim)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)

    cone.mats = [[Matrix{Any}(undef, size(Pk, 2), size(Pk, 2)) for _ in 1:(R - 1)] for Pk in cone.Ps]
    cone.matfact = [[cholesky(hcat([one(T)])) for _ in 1:R] for _ in cone.Ps]
    cone.Λi_Λ = [Vector{Matrix{T}}(undef, R - 1) for Psk in Ps]
    @inbounds for k in eachindex(Ps), r in 1:(R - 1)
        cone.Λi_Λ[k][r] = similar(cone.grad, size(Ps[k], 2), size(Ps[k], 2))
    end
    cone.Λ11 = [similar(cone.grad, size(Psk, 2), size(Psk, 2)) for Psk in Ps]
    cone.tmpΛ11 = [similar(cone.grad, size(Psk, 2), size(Psk, 2)) for Psk in Ps]
    cone.tmpLL = [similar(cone.grad, size(Psk, 2), size(Psk, 2)) for Psk in Ps]
    cone.tmpLU = [similar(cone.grad, size(Psk, 2), U) for Psk in Ps]
    cone.tmpLU2 = [similar(cone.grad, size(Psk, 2), U) for Psk in Ps]
    cone.tmpUU_vec = [similar(cone.grad, U, U) for _ in eachindex(Ps)]
    cone.tmpUU = similar(cone.grad, U, U)
    cone.PΛiPs1 = [Vector{Matrix{T}}(undef, R) for Psk in Ps]
    cone.PΛiPs2 = [Vector{Matrix{T}}(undef, R) for Psk in Ps]
    @inbounds for k in eachindex(Ps), r in 1:(R - 1)
        cone.PΛiPs1[k][r] = similar(cone.grad, U, U)
        cone.PΛiPs2[k][r] = similar(cone.grad, U, U)
    end
    cone.lambdafact = Vector{Any}(undef, length(Ps))
    cone.point_views = [view(cone.point, block_idxs(U, i)) for i in 1:R]
    return
end

get_nu(cone::WSOSInterpEpiNormInf) = cone.R * sum(size(Pk, 2) for Pk in cone.Ps)

function set_initial_point(arr::AbstractVector, cone::WSOSInterpEpiNormInf)
    arr[1:cone.U] .= 1
    arr[(cone.U + 1):end] .= 0
    return arr
end

# function update_feas(cone::WSOSInterpEpiNormInf)
#     @assert !cone.feas_updated
#     U = cone.U
#     point = cone.point
#
#     # cone.is_feas = true
#     # @inbounds for k in eachindex(cone.Ps)
#     #     P = cone.Ps[k]
#     #     lambda_1 = Symmetric(P' * Diagonal(point[1:U]) * P)
#     #     fact_1 = cholesky(lambda_1, check = false)
#     #     if isposdef(fact_1)
#     #         for i in 2:cone.R
#     #             lambda_i = Symmetric(P' * Diagonal(point[block_idxs(U, i)]) * P)
#     #             LL = fact_1.L \ lambda_i
#     #             if !isposdef(lambda_1 - LL' * LL)
#     #                 cone.is_feas = false
#     #                 break
#     #             end
#     #         end
#     #     else
#     #         cone.is_feas = false
#     #         break
#     #     end
#     # end
#
#     cone.is_feas = true
#     @inbounds for k in eachindex(cone.Ps)
#         P = cone.Ps[k]
#         lambda_1 = Symmetric(P' * Diagonal(point[1:U]) * P)
#         fact_1 = cholesky(lambda_1, check = false)
#         if isposdef(fact_1)
#             for i in 2:cone.R
#                 lambda_i = Symmetric(P' * Diagonal(point[block_idxs(U, i)]) * P)
#                 if !isposdef(lambda_1 - lambda_i) || !isposdef(lambda_1 + lambda_i)
#                     cone.is_feas = false
#                     break
#                 end
#             end
#         else
#             cone.is_feas = false
#             break
#         end
#     end
#
#     cone.feas_updated = true
#     return cone.is_feas
# end

function update_feas(cone::WSOSInterpEpiNormInf)
    @assert !cone.feas_updated
    U = cone.U
    R = cone.R
    lambdafact = cone.lambdafact
    point_views = cone.point_views

    cone.is_feas = true
    @inbounds for k in eachindex(cone.Ps)
        Psk = cone.Ps[k]
        Λ11j = cone.Λ11[k]
        tmpΛ11j = cone.tmpΛ11[k]
        LLk = cone.tmpLL[k]
        LUk = cone.tmpLU[k]
        Λi_Λ = cone.Λi_Λ[k]
        matsk = cone.mats[k]
        factk = cone.matfact[k]

        # first lambda
        @. LUk = Psk' * point_views[1]'
        mul!(tmpΛ11j, LUk, Psk)
        copyto!(Λ11j, tmpΛ11j)
        lambdafact[k] = cholesky!(Symmetric(tmpΛ11j, :U), check = false)
        if !isposdef(lambdafact[k])
            cone.is_feas = false
            break
        end

        uo = U + 1
        @inbounds for r in 2:R
            r1 = r - 1
            matr = matsk[r1]
            factr = factk[r1]
            @. LUk = Psk' * point_views[r]'
            mul!(LLk, LUk, Psk)

            # not using lambdafact.L \ lambda with an syrk because storing lambdafact \ lambda is useful later
            ldiv!(Λi_Λ[r1], lambdafact[k], LLk)
            copyto!(matr, Λ11j)
            mul!(matr, LLk, Λi_Λ[r1], -1, 1)

            # ldiv!(lambdafact[k].L, LLk)
            # mat = Λ11j - LLk' * LLk
            factk[r1] = cholesky!(Symmetric(matr, :U), check = false)
            if !isposdef(factk[r1])
                cone.is_feas = false
                cone.feas_updated = true
                return cone.is_feas
            end
            uo += U
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(cone::WSOSInterpEpiNormInf) = true

# TODO common code could be refactored with epinormeucl version
function update_grad(cone::WSOSInterpEpiNormInf)
    # cone.grad = ForwardDiff.gradient(cone.barrier, cone.point)

    @assert cone.is_feas
    U = cone.U
    R = cone.R
    R2 = R - 2
    lambdafact = cone.lambdafact
    matfact = cone.matfact

    cone.grad .= 0
    @inbounds for k in eachindex(cone.Ps)
        Psk = cone.Ps[k]
        LUk = cone.tmpLU[k]
        LUk2 = cone.tmpLU2[k]
        UUk = cone.tmpUU_vec[k]
        PΛiPs1 = cone.PΛiPs1[k]
        PΛiPs2 = cone.PΛiPs2[k]
        Λi_Λ = cone.Λi_Λ[k]

        # P * inv(Λ_11) * P' for (1, 1) hessian block and adding to PΛiPs[r][r]
        ldiv!(LUk, cone.lambdafact[k].L, Psk')
        mul!(UUk, LUk', LUk)

        # prep PΛiPs
        # get all the PΛiPs that are in row one or on the diagonal
        @inbounds for r in 1:(R - 1)
            # block-(1,1) is P * inv(mat) * P'
            ldiv!(LUk, matfact[k][r].L, Psk')
            mul!(PΛiPs1[r], LUk', LUk)
            # block (1,2)
            ldiv!(LUk, matfact[k][r], Psk')
            mul!(LUk2, Λi_Λ[r], LUk)
            mul!(PΛiPs2[r], Psk, LUk2, -1, false)
        end

        # (1, 1)-block
        # gradient is diag of sum(-PΛiPs[i][i] for i in 1:R) + (R - 1) * Lambda_11 - Lambda_11
        @inbounds for i in 1:U
            cone.grad[i] += UUk[i, i] * R2
            @inbounds for r in 1:(R - 1)
                cone.grad[i] -= PΛiPs1[r][i, i] * 2
            end
        end
        idx = U + 1
        @inbounds for r in 1:(R - 1), i in 1:U
            cone.grad[idx] -= 2 * PΛiPs2[r][i, i]
            idx += 1
        end
    end # j
    # # @show cone.grad ./ fd_grad

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSInterpEpiNormInf)
    @timeit cone.timer "hess" begin
    # cone.hess.data .= ForwardDiff.hessian(cone.barrier, cone.point)
    # fd_hess = ForwardDiff.hessian(cone.barrier, cone.point)

    @assert cone.grad_updated
    U = cone.U
    R = cone.R
    R2 = R - 2
    hess = cone.hess.data
    UU = cone.tmpUU
    matfact = cone.matfact

    hess .= 0
    @inbounds for k in eachindex(cone.Ps)
        PΛiPs1 = cone.PΛiPs1[k]
        PΛiPs2 = cone.PΛiPs2[k]
        UUk = cone.tmpUU_vec[k]

        @. hess[1:U, 1:U] -= abs2(UUk) * R2
        @inbounds for r in 1:(R - 1)
            # @. UU = abs2(PΛiPs2[r])
            @. UU = PΛiPs2[r] * PΛiPs2[r]' # TODO be more careful with numeric for PΛiPs2[r]
            @. UU += abs2(PΛiPs1[r])
            @. hess[1:U, 1:U] += UU
            @. hess[1:U, 1:U] += UU
            # blocks (r, r)
            idxs = block_idxs(U, r + 1)
            @. hess[idxs, idxs] += UU
            # blocks (1,r)
            @. UU = PΛiPs2[r] * PΛiPs1[r]
            @. UU += UU'
            @. hess[1:U, idxs] += UU
        end
    end

    @. hess[:, (U + 1):cone.dim] *= 2
    # # @show cone.hess ./ fd_hess

    cone.hess_updated = true
    end # timer
    return cone.hess
end

# function update_hess(cone::WSOSInterpEpiNormInf)
#     # cone.hess.data .= ForwardDiff.hessian(cone.barrier, cone.point)
#     # fd_hess = ForwardDiff.hessian(cone.barrier, cone.point)
#
#     @assert cone.grad_updated
#     U = cone.U
#     R = cone.R
#     R2 = R - 2
#     hess = cone.hess.data
#     UU = cone.tmpUU
#     matfact = cone.matfact
#
#     hess .= 0
#     @inbounds for k in eachindex(cone.Ps)
#         Psk = cone.Ps[k]
#         PΛiPs1 = cone.PΛiPs1[k]
#         PΛiPs2 = cone.PΛiPs2[k]
#         Λi_Λ = cone.Λi_Λ[k]
#         UUk = cone.tmpUU_vec[k]
#         LUk = cone.tmpLU[k]
#         LUk2 = cone.tmpLU2[k]
#
#         @inbounds for i in 1:U, k in 1:i
#             hess[k, i] -= abs2(UUk[k, i]) * R2
#         end
#
#         @inbounds for r in 1:(R - 1)
#             @. hess[1:U, 1:U] += abs2(PΛiPs1[r])
#             idxs = block_idxs(U, r + 1)
#             # block (1,1)
#             @. UU = abs2(PΛiPs2[r])
#             # safe to ovewrite UUk now
#             @. UUk = UU + UU'
#             @. hess[1:U, 1:U] += UUk
#             @. hess[1:U, 1:U] += abs2(PΛiPs1[r])
#             # blocks (1,r)
#             @. hess[1:U, idxs] += PΛiPs2[r] * PΛiPs1[r] + PΛiPs1[r] * PΛiPs2[r]
#
#             # blocks (r, r2)
#             # NOTE for hess[idxs, idxs], UU and UUk are symmetric
#             @. UU = PΛiPs2[r] * PΛiPs2[r]'
#             @. UUk = abs2(PΛiPs1[r])
#             @. hess[idxs, idxs] += UU + UUk
#         end
#     end
#     @. hess[:, (U + 1):cone.dim] *= 2
#     # @show cone.hess ./ fd_hess
#
#     cone.hess_updated = true
#     return cone.hess
# end

use_correction(::WSOSInterpEpiNormInf) = false
