#=
Copyright 2018, Chris Coey and contributors

# TODO can parallelize expensive functions mapped over primitive cones
=#

# cone object
abstract type PrimitiveCone end

# TODO order primitive cones so easiest ones to check incone are first
mutable struct Cone
    prmtvs::Vector{PrimitiveCone}
    idxs::Vector{UnitRange{Int}}

    function Cone(prmtvs::Vector{<:PrimitiveCone}, idxs::Vector{UnitRange{Int}})
        cone = new()
        @assert length(prmtvs) == length(idxs)
        cone.prmtvs = prmtvs
        cone.idxs = idxs
        return cone
    end
end
Cone() = Cone(PrimitiveCone[], UnitRange{Int}[])

function addprimitivecone!(
    cone::Cone,
    prmtv::PrimitiveCone,
    idx::UnitRange{Int},
    )
    @assert dimension(prmtv) == length(idx)
    push!(cone.prmtvs, prmtv)
    push!(cone.idxs, idx)
    return cone
end

# calculate complexity parameter of the barrier (sum of the primitive cone barrier parameters)
barrierpar(cone::Cone)::Float64 = (isempty(cone.prmtvs) ? 0.0 : sum(barrierpar_prmtv(prmtv) for prmtv in cone.prmtvs))

function loadpnt!(cone::Cone, ts::Vector{Float64}, tz::Vector{Float64})
    for k in eachindex(cone.prmtvs)
        (v1, v2) = (cone.prmtvs[k].usedual ? (ts, tz) : (tz, ts))
        loadpnt_prmtv!(cone.prmtvs[k], view(v2, cone.idxs[k]))
    end
    return nothing
end

incone(cone::Cone, scal::Float64) = all(incone_prmtv(cone.prmtvs[k], scal) for k in eachindex(cone.prmtvs))

function getinitsz!(ts, tz, cone)
    for k in eachindex(cone.prmtvs)
        (v1, v2) = (cone.prmtvs[k].usedual ? (ts, tz) : (tz, ts))
        getintdir_prmtv!(view(v2, cone.idxs[k]), cone.prmtvs[k])
        @assert incone_prmtv(cone.prmtvs[k], 1.0)
        calcg_prmtv!(view(v1, cone.idxs[k]), cone.prmtvs[k])
        @. @views v1[cone.idxs[k]] *= -1.0
    end
    return (ts, tz)
end

function calcg!(g::Vector{Float64}, cone::Cone)
    for k in eachindex(cone.prmtvs)
        calcg_prmtv!(view(g, cone.idxs[k]), cone.prmtvs[k])
    end
    return g
end

# calculate neighborhood distance to central path
function calcnbhd!(g, ts, tz, mu, cone)
    for k in eachindex(cone.prmtvs)
        calcg_prmtv!(view(g, cone.idxs[k]), cone.prmtvs[k])
        (v1, v2) = (cone.prmtvs[k].usedual ? (ts, tz) : (tz, ts))
        @. @views v1[cone.idxs[k]] += mu*g[cone.idxs[k]]
        # @. @views v1[cone.idxs[k]] += g[cone.idxs[k]]
        calcHiarr_prmtv!(view(v2, cone.idxs[k]), view(v1, cone.idxs[k]), cone.prmtvs[k])
    end
    return dot(ts, tz)
end

# utilities for converting between smat and svec forms (lower triangle) for symmetric matrices
# TODO only need to do lower triangle if use symmetric matrix types
const rt2 = sqrt(2)
const rt2i = inv(rt2)

function mattovec!(vec::AbstractVector, mat::AbstractMatrix)
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        if i == j
            vec[k] = mat[i,j]
        else
            vec[k] = rt2*mat[i,j]
        end
        k += 1
    end
    return vec
end

function vectomat!(mat::AbstractMatrix, vec::AbstractVector)
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        if i == j
            mat[i,j] = vec[k]
        else
            mat[i,j] = mat[j,i] = rt2i*vec[k]
        end
        k += 1
    end
    return mat
end


# common primitive cone functions

function factH(prmtv::PrimitiveCone)
    @. prmtv.H2 = prmtv.H

    prmtv.F = bunchkaufman!(Symmetric(prmtv.H2, :U), true, check=false)
    return issuccess(prmtv.F)

    # prmtv.F = cholesky!(Symmetric(prmtv.H2), Val(true), check=false)
    # if !isposdef(prmtv.F)
    #     println("primitive cone Hessian was singular")
    #     @. prmtv.H2 = prmtv.H
    #     prmtv.F = PositiveFactorizations.cholesky!(PositiveFactorizations.Positive, prmtv.H2)
    # end
    # return true
end

# calcg_prmtv!(g::AbstractVector{Float64}, prmtv::PrimitiveCone) = (@. g = prmtv.g; g)
# calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::PrimitiveCone) = ldiv!(prod, prmtv.F, arr)
# calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::PrimitiveCone) = mul!(prod, prmtv.H, arr)

# calcg_prmtv!(g::AbstractVector{Float64}, prmtv::PrimitiveCone) = (@. g = prmtv.g; lmul!(prmtv.iscal, g); g)
# calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::PrimitiveCone) = (ldiv!(prod, prmtv.F, arr); lmul!(prmtv.scal, prod); lmul!(prmtv.scal, prod); prod)
# calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::PrimitiveCone) = (mul!(prod, prmtv.H, arr); lmul!(prmtv.iscal, prod); lmul!(prmtv.iscal, prod); prod)
