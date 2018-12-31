#=
Copyright 2018, Chris Coey and contributors

epigraph of Euclidean (2-)norm (AKA second-order cone)
(u in R, w in R^n) : u >= norm_2(w)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-log(u^2 - norm(w)^2)
=#

mutable struct EpiNormEucl <: PrimitiveCone
    usedual::Bool
    dim::Int
    pnt::AbstractVector{Float64}
    dist::Float64
    Hi::Matrix{Float64}
    H::Matrix{Float64}

    function EpiNormEucl(dim::Int, isdual::Bool)
        prmtv = new()
        prmtv.usedual = isdual
        prmtv.dim = dim
        prmtv.Hi = Matrix{Float64}(undef, dim, dim)
        prmtv.H = similar(prmtv.Hi)
        return prmtv
    end
end

EpiNormEucl(dim::Int) = EpiNormEucl(dim, false)

dimension(prmtv::EpiNormEucl) = prmtv.dim
barrierpar_prmtv(prmtv::EpiNormEucl) = 1
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::EpiNormEucl) = (@. arr = 0.0; arr[1] = 1.0; arr)
loadpnt_prmtv!(prmtv::EpiNormEucl, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::EpiNormEucl, scal::Float64)
    u = prmtv.pnt[1]
    w = view(prmtv.pnt, 2:prmtv.dim)
    if u <= 0.0
        return false
    end
    dist = abs2(u) - sum(abs2, w)
    if dist <= 0.0
        return false
    end
    prmtv.dist = dist

    H = prmtv.H
    Hi = prmtv.Hi
    mul!(Hi, prmtv.pnt, prmtv.pnt')
    @. Hi += Hi
    Hi[1,1] -= dist
    for j in 2:prmtv.dim
        Hi[j,j] += dist
    end
    @. H = Hi
    for j in 2:prmtv.dim
        H[1,j] = H[j,1] = -H[j,1]
    end
    @. H *= abs2(inv(dist))
    return true
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::EpiNormEucl) = (@. g = prmtv.pnt/prmtv.dist; g[1] = -g[1]; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr, prmtv::EpiNormEucl) = mul!(prod, prmtv.Hi, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr, prmtv::EpiNormEucl) = mul!(prod, prmtv.H, arr)
