#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

interpolation-based weighted-sum-of-squares (multivariate) polynomial cone parametrized by interpolation points Ps

definition and dual barrier from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792

TODO
- can perform loop for calculating g and H in parallel
- scale the interior direction
- check if gradient and hessian are correct for complex case
=#

RealOrComplexF64 = Union{Float64, ComplexF64}

mutable struct WSOSPolyInterp{T <: RealOrComplexF64} <: Cone
    use_dual::Bool
    dim::Int
    Ps::Vector{Matrix{T}}

    point::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    Hi::Matrix{Float64}
    F # TODO prealloc
    tmpLL::Vector{Matrix{T}}
    tmpUL::Vector{Matrix{T}}
    tmpLU::Vector{Matrix{T}}
    tmpUU::Matrix{T}
    ΛFs::Vector{CholeskyPivoted{T, Matrix{T}}}

    function WSOSPolyInterp(dim::Int, Ps::Vector{Matrix{T}}, is_dual::Bool) where {T <: RealOrComplexF64}
        for k in eachindex(Ps)
            @assert size(Ps[k], 1) == dim
        end
        cone = new{T}()
        cone.use_dual = !is_dual # using dual barrier
        cone.dim = dim
        cone.Ps = Ps
        return cone
    end
end

WSOSPolyInterp(dim::Int, Ps::Vector{Matrix{T}}) where {T <: RealOrComplexF64} = WSOSPolyInterp{T}(dim, Ps, false)

function setup_data(cone::WSOSPolyInterp{T}) where T
    dim = cone.dim
    cone.g = Vector{Float64}(undef, dim)
    cone.H = similar(cone.g, dim, dim)
    cone.H2 = similar(cone.H)
    cone.Hi = similar(cone.H)
    Ps = cone.Ps
    cone.tmpLL = [Matrix{T}(undef, size(Pk, 2), size(Pk, 2)) for Pk in Ps]
    cone.tmpUL = [Matrix{T}(undef, dim, size(Pk, 2)) for Pk in Ps]
    cone.tmpLU = [Matrix{T}(undef, size(Pk, 2), dim) for Pk in Ps]
    cone.tmpUU = Matrix{T}(undef, dim, dim)
    cone.ΛFs = Vector{CholeskyPivoted{T, Matrix{T}}}(undef, length(Ps))
    return
end

get_nu(cone::WSOSPolyInterp) = sum(size(Pk, 2) for Pk in cone.Ps)

set_initial_point(arr::AbstractVector{Float64}, cone::WSOSPolyInterp) = (@. arr = 1.0; arr)

_AtA!(U::Matrix{T}, A::Matrix{T}) where {T <: Real} = BLAS.syrk!('U', 'T', one(T), A, zero(T), U)
_AtA!(U::Matrix{Complex{T}}, A::Matrix{Complex{T}}) where {T <: Real} = BLAS.herk!('U', 'C', one(T), A, zero(T), U)

function check_in_cone(cone::WSOSPolyInterp; invert_hess::Bool = true)
    Ps = cone.Ps
    LLs = cone.tmpLL
    ULs = cone.tmpUL
    LUs = cone.tmpLU
    UU = cone.tmpUU
    ΛFs = cone.ΛFs
    D = Diagonal(cone.point)

    # @timeit Hypatia.to "buildlam" begin

    for k in eachindex(Ps) # TODO can be done in parallel
        # Λ = Pk' * Diagonal(point) * Pk
        # TODO LDLT calculation could be faster
        # TODO mul!(A, B', Diagonal(x)) calls extremely inefficient method but don't need ULk
        Pk = Ps[k]
        ULk = ULs[k]
        LLk = LLs[k]
        # @timeit Hypatia.to "UL"
        mul!(ULk, D, Pk)
        # @timeit Hypatia.to "LL"
        mul!(LLk, Pk', ULk)

        # pivoted cholesky and triangular solve method
        # @timeit Hypatia.to "cholesky"
        ΛFk = cholesky!(Hermitian(LLk, :L), Val(true), check = false)
        if !isposdef(ΛFk)
            return false
        end
        ΛFs[k] = ΛFk
    end

    # end # incone check
    # @timeit Hypatia.to "gradhess" begin

    g = cone.g
    H = cone.H
    @. g = 0.0
    @. H = 0.0
    # UUd = view(UU, diagind(UU))
    # UUH = Hermitian(UU, :U)

    for k in eachindex(Ps) # TODO can be done in parallel, but need multiple tmp3s
        LUk = LUs[k]
        ΛFk = ΛFs[k]
        @timeit Hypatia.to "view" LUk .= view(Ps[k]', ΛFk.p, :)
        @timeit Hypatia.to "ldiv" ldiv!(LowerTriangular(ΛFk.L), LUk) # TODO check calls best triangular solve
        @timeit Hypatia.to "AtA" _AtA!(UU, LUk)

        # LUk = LUs[k]
        # ΛFk = ΛFs[k]
        # ULk = ULs[k]
        # # @timeit Hypatia.to "view"
        # LUk .= view(Ps[k]', ΛFk.p, :)
        # # @timeit Hypatia.to "ldiv"
        # ldiv!(LowerTriangular(ΛFk.L), LUk) # TODO check calls best triangular solve
        # # @timeit Hypatia.to "AtA"
        # _AtA!(UU, LUk)

        # LAPACK.trtri!('U', 'N', ΛFk.factors)
        # ULk .= view(Ps[k], :, ΛFk.p)
        # BLAS.trmm!('R', 'U', 'N', 'N', 1.0, ΛFk.factors, ULk)
        # ULk = ULk * UpperTriangular(ΛFk.factors)
        # UU = ULk * ULk'

        # U_i = inv(ΛFk.U)
        # # LAPACK.trtri!('U', 'N', ΛFk.factors)
        # ULk .= view(Ps[k], :, ΛFk.p)
        # ULk = ULk * U_i
        # UU = ULk * ULk'

        # lambda_i = inv(ΛFk)
        # mul!(LUk, lambda_i, Ps[k]')
        # mul!(UU, Ps[k], LUk)

        # @timeit Hypatia.to "g" @. g -= real(UUd)
        # @timeit Hypatia.to "H" @. H += abs2(UU)

        @timeit Hypatia.to "gH" begin
        for j in eachindex(g)
            @inbounds g[j] -= real(UU[j, j])
            @simd for i in 1:j
                @inbounds H[i, j] += abs2(UU[i, j])
            end
        end
        end
    end
    # end # timeit gradhess
    # @assert -dot(cone.point, cone.g) ≈ get_nu(cone) atol=1e-3 rtol=1e-3
    # if norm(Symmetric(cone.H, :U) * cone.point + cone.g) > 1e-3
    #     @show Symmetric(cone.H, :U) * cone.point + cone.g
    # end
    # @assert Symmetric(cone.H, :U) * cone.point ≈ -cone.g atol=1e-1 rtol=1e-1

    # if invert_hess
    #     return factorize_hess(cone)
    # else
        return true
    # end
end
