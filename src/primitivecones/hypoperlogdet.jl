#=
Copyright 2018, Chris Coey and contributors

(closure of) hypograph of perspective of (natural) log of determinant of a (row-wise lower triangle i.e. svec space) symmetric positive define matrix
(smat space) (u in R, v in R_+, w in S_+) : u <= v*logdet(W/v)
(see equivalent MathOptInterface LogDetConeConeTriangle definition)

barrier (guessed, based on analogy to hypoperlog barrier)
-log(v*logdet(W/v) - u) - logdet(W) - log(v)

TODO only use one decomposition on Symmetric(W) for isposdef and logdet
TODO symbolically calculate gradient and Hessian
=#

mutable struct HypoPerLogdet <: PrimitiveCone
    usedual::Bool
    dim::Int
    side::Int
    pnt::AbstractVector{Float64}
    mat::Matrix{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function HypoPerLogdet(dim::Int, isdual::Bool)
        prmtv = new()
        prmtv.usedual = isdual
        prmtv.dim = dim
        side = round(Int, sqrt(0.25 + 2*(dim - 2)) - 0.5)
        prmtv.side = side
        prmtv.mat = Matrix{Float64}(undef, side, side)
        prmtv.g = Vector{Float64}(undef, dim)
        prmtv.H = similar(prmtv.g, dim, dim)
        prmtv.H2 = similar(prmtv.H)
        function barfun(pnt)
            u = pnt[1]
            v = pnt[2]
            W = similar(pnt, side, side)
            vectomat!(W, view(pnt, 3:dim))
            return -log(v*logdet(W/v) - u) - logdet(W) - log(v)
        end
        prmtv.barfun = barfun
        prmtv.diffres = DiffResults.HessianResult(prmtv.g)
        return prmtv
    end
end

HypoPerLogdet(dim::Int) = HypoPerLogdet(dim, false)

dimension(prmtv::HypoPerLogdet) = prmtv.dim
barrierpar_prmtv(prmtv::HypoPerLogdet) = prmtv.side + 2

function getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::HypoPerLogdet)
    arr[1] = -1.0
    arr[2] = 1.0
    mattovec!(view(arr, 3:prmtv.dim), Matrix(1.0I, prmtv.side, prmtv.side))
    return arr
end

loadpnt_prmtv!(prmtv::HypoPerLogdet, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::HypoPerLogdet, scal::Float64)
    pnt = prmtv.pnt/scal
    u = pnt[1]
    v = pnt[2]
    W = prmtv.mat
    vectomat!(W, view(pnt, 3:prmtv.dim))
    if v <= 0.0 || !isposdef(Symmetric(W)) || u >= v*logdet(Symmetric(W)/v) # TODO only use one decomposition on Symmetric(W) for isposdef and logdet
        return false
    end

    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    prmtv.diffres = ForwardDiff.hessian!(prmtv.diffres, prmtv.barfun, pnt)
    prmtv.g .= DiffResults.gradient(prmtv.diffres)
    prmtv.H .= DiffResults.hessian(prmtv.diffres)

    return factH(prmtv)
end
