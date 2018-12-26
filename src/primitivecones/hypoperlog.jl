#=
Copyright 2018, Chris Coey and contributors

(closure of) hypograph of perspective of (natural) log (AKA exponential cone)
(u in R, v in R_+, w in R_+) : u <= v*log(w/v)

barrier from "A homogeneous interior-point algorithm for nonsymmetric convex conic optimization" by Skajaa & Ye 2014
-log(v*log(w/v) - u) - log(w) - log(v)

TODO allow different log bases?
TODO maybe use StaticArrays
TODO try to extend to case w in R^n
TODO could write the inverse hessian analytically rather than factorizing
TODO choose a better interior direction
=#

mutable struct HypoPerLog <: PrimitiveCone
    usedual::Bool
    pnt::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F

    function HypoPerLog(isdual::Bool)
        prmtv = new()
        prmtv.usedual = isdual
        prmtv.g = Vector{Float64}(undef, 3)
        prmtv.H = similar(prmtv.g, 3, 3)
        prmtv.H2 = similar(prmtv.H)
        return prmtv
    end
end

HypoPerLog() = HypoPerLog(false)

dimension(prmtv::HypoPerLog) = 3
barrierpar_prmtv(prmtv::HypoPerLog) = 3
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::HypoPerLog) = (arr[1] = -1.0; arr[2] = 1.0; arr[3] = 1.0; arr)
loadpnt_prmtv!(prmtv::HypoPerLog, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::HypoPerLog, scal::Float64)
    @show prmtv.pnt
    @show scal

    u = prmtv.pnt[1]/scal; v = prmtv.pnt[2]/scal; w = prmtv.pnt[3]/scal
    if (v <= 0.0) || (w <= 0.0)
        return false
    end
    lwv = log(w/v)
    vlwv = v*lwv
    vlwvu = vlwv - u
    @show vlwvu
    if vlwvu <= 0.0
        return false
    end

    # gradient
    ivlwvu = inv(vlwvu)
    g = prmtv.g
    g[1] = ivlwvu
    g[2] = ivlwvu*(v - u - 2.0*vlwvu)/v
    g[3] = -(1.0 + v*ivlwvu)/w

    @show g

    # Hessian
    vw = v/w
    ivlwvu2 = abs2(ivlwvu)
    H = prmtv.H
    H[1,1] = ivlwvu2
    H[1,2] = H[2,1] = -(lwv - 1.0)*ivlwvu2
    H[1,3] = H[3,1] = -vw*ivlwvu2
    H[2,2] = abs2(lwv - 1.0)*ivlwvu2 + ivlwvu/v + inv(abs2(v))
    H[2,3] = H[3,2] = vw*(lwv - 1.0)*ivlwvu2 - ivlwvu/w
    H[3,3] = abs2(vw)*ivlwvu2 + vw/w*ivlwvu + inv(abs2(w))

    @show H

    return factH(prmtv)
end
