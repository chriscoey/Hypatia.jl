#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

interpolation-based weighted-sum-of-squares (multivariate) polynomial cone parametrized by interpolation points ipwt

definition and dual barrier from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792

TODO can perform loop for calculating g and H in parallel
TODO maybe can avoid final factorization?
TODO scale the interior direction
=#

mutable struct WSOSPolyInterp <: PrimitiveCone
    usedual::Bool
    dim::Int
    ipwt::Vector{Matrix{Float64}}
    pnt::AbstractVector{Float64}
    scalpnt::Vector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F # TODO prealloc
    tmp1::Vector{Matrix{Float64}}
    tmp2::Vector{Matrix{Float64}}
    tmp3::Matrix{Float64}
    scal::Float64
    # d::Vector{POSVXData}

    function WSOSPolyInterp(dim::Int, ipwt::Vector{Matrix{Float64}}, isdual::Bool)
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == dim
        end
        prmtv = new()
        prmtv.usedual = !isdual # using dual barrier
        prmtv.dim = dim
        prmtv.ipwt = ipwt
        prmtv.scalpnt = similar(ipwt[1], dim)
        prmtv.g = similar(ipwt[1], dim)
        prmtv.H = similar(ipwt[1], dim, dim)
        prmtv.H2 = similar(prmtv.H)
        prmtv.tmp1 = [similar(ipwt[1], size(ipwtj, 2), size(ipwtj, 2)) for ipwtj in ipwt]
        prmtv.tmp2 = [similar(ipwt[1], size(ipwtj, 2), dim) for ipwtj in ipwt]
        prmtv.tmp3 = similar(ipwt[1], dim, dim)
        # prmtv.d = [POSVXData(prmtv.tmp1[j], prmtv.tmp2[j]) for j in eachindex(ipwt)]
        return prmtv
    end
end

WSOSPolyInterp(dim::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterp(dim, ipwt, false)

dimension(prmtv::WSOSPolyInterp) = prmtv.dim
barrierpar_prmtv(prmtv::WSOSPolyInterp) = sum(size(ipwtj, 2) for ipwtj in prmtv.ipwt)
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::WSOSPolyInterp) = (@. arr = 1.0; arr)
loadpnt_prmtv!(prmtv::WSOSPolyInterp, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::WSOSPolyInterp, scal::Float64)
    prmtv.scal = scal
    @. prmtv.scalpnt = prmtv.pnt/prmtv.scal

    @. prmtv.g = 0.0
    @. prmtv.H = 0.0
    tmp3 = prmtv.tmp3

    for j in eachindex(prmtv.ipwt) # TODO can be done in parallel, but need multiple tmp3s
        ipwtj = prmtv.ipwt[j]
        tmp1j = prmtv.tmp1[j]
        tmp2j = prmtv.tmp2[j]

        # tmp1j = ipwtj'*Diagonal(pnt)*ipwtj
        # mul!(tmp2j, ipwtj', Diagonal(prmtv.scalpnt)) # TODO dispatches to an extremely inefficient method
        @. tmp2j = ipwtj' * prmtv.scalpnt'
        mul!(tmp1j, tmp2j, ipwtj)

        # pivoted cholesky and triangular solve method
        F = cholesky!(Symmetric(tmp1j, :L), Val(true), check=false)
        if !isposdef(F)
            return false
        end

        tmp2j .= view(ipwtj', F.p, :)
        ldiv!(F.L, tmp2j) # TODO make sure calls best triangular solve
        # mul!(tmp3, tmp2j', tmp2j)
        BLAS.syrk!('U', 'T', 1.0, tmp2j, 0.0, tmp3)

        # posvx solve method
        # tmp2j .= ipwtj' # TODO eliminate by transposing in construction
        # PDPiP = similar(tmp2j)
        # success = hypatia_posvx!(PDPiP, tmp1j, tmp2j, prmtv.d[j])
        # if !success
        #     return false
        # end
        # mul!(tmp3, ipwtj, PDPiP)

        @inbounds for j in eachindex(prmtv.g)
            prmtv.g[j] -= tmp3[j,j]
            @inbounds for i in 1:j
                prmtv.H[i,j] += abs2(tmp3[i,j])
            end
        end
    end

    return factH(prmtv)
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::WSOSPolyInterp) = (@. g = prmtv.g/prmtv.scal; g)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr, prmtv::WSOSPolyInterp) = (mul!(prod, Symmetric(prmtv.H), arr); @. prod = prod / prmtv.scal / prmtv.scal; prod)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr, prmtv::WSOSPolyInterp) = (ldiv!(prod, prmtv.F, arr); @. prod = prod * prmtv.scal * prmtv.scal; prod)
calcHarr_prmtv!(arr::AbstractArray{Float64}, prmtv::WSOSPolyInterp) = (lmul!(Symmetric(prmtv.H), arr); @. arr = arr / prmtv.scal / prmtv.scal; arr)
calcHiarr_prmtv!(arr::AbstractArray{Float64}, prmtv::WSOSPolyInterp) = (ldiv!(prmtv.F, arr); @. arr = arr * prmtv.scal * prmtv.scal; arr)
