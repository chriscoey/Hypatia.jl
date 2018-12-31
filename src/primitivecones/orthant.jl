#=
Copyright 2018, Chris Coey and contributors

nonnegative/nonpositive orthant cones
nonnegative cone: w in R^n : w_i >= 0
nonpositive cone: w in R^n : w_i <= 0

barriers from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
nonnegative cone: -sum_i(log(u_i))
nonpositive cone: -sum_i(log(-u_i))
=#

mutable struct Nonnegative <: PrimitiveCone
    usedual::Bool
    dim::Int
    pnt::AbstractVector{Float64}
    invpnt::Vector{Float64}

    function Nonnegative(dim::Int, isdual::Bool)
        prmtv = new()
        prmtv.usedual = isdual
        prmtv.dim = dim
        prmtv.invpnt = Vector{Float64}(undef, dim)
        return prmtv
    end
end

Nonnegative(dim::Int) = Nonnegative(dim, false)
Nonnegative() = Nonnegative(1)

mutable struct Nonpositive <: PrimitiveCone
    usedual::Bool
    dim::Int
    pnt::AbstractVector{Float64}
    invpnt::Vector{Float64}

    function Nonpositive(dim::Int, isdual::Bool)
        prmtv = new()
        prmtv.usedual = isdual
        prmtv.dim = dim
        prmtv.invpnt = Vector{Float64}(undef, dim)
        return prmtv
    end
end

Nonpositive(dim::Int) = Nonpositive(dim, false)
Nonpositive() = Nonpositive(1)

OrthantCone = Union{Nonnegative, Nonpositive}

dimension(prmtv::OrthantCone) = prmtv.dim
barrierpar_prmtv(prmtv::OrthantCone) = prmtv.dim

getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::Nonnegative) = (@. arr = 1.0; arr)
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::Nonpositive) = (@. arr = -1.0; arr)

loadpnt_prmtv!(prmtv::OrthantCone, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

incone_prmtv(prmtv::Nonnegative, scal::Float64) = all(u -> (u > 0.0), prmtv.pnt)
incone_prmtv(prmtv::Nonpositive, scal::Float64) = all(u -> (u < 0.0), prmtv.pnt)

function calcg_prmtv!(g::AbstractVector{Float64}, prmtv::OrthantCone)
    @. prmtv.invpnt = inv(prmtv.pnt)
    @. g = -prmtv.invpnt
    return g
end

calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::UniformScaling{Float64}, prmtv::OrthantCone) = (@. prod = abs2(prmtv.invpnt) * arr.λ; prod)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::OrthantCone) = (@. prod = abs2(prmtv.invpnt) * arr; prod)
calcHarr_prmtv!(arr::AbstractArray{Float64}, prmtv::OrthantCone) = (@. arr = abs2(prmtv.invpnt) * arr; arr)

calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::UniformScaling{Float64}, prmtv::OrthantCone) = (@. prod = abs2(prmtv.pnt) / arr.λ; prod)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::OrthantCone) = (@. prod = abs2(prmtv.pnt) * arr; prod)
calcHiarr_prmtv!(arr::AbstractArray{Float64}, prmtv::OrthantCone) = (@. arr = abs2(prmtv.pnt) * arr; arr)
