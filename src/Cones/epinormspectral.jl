#=
Copyright 2018, Chris Coey and contributors

epigraph of matrix spectral norm (operator norm associated with standard Euclidean norm; i.e. maximum singular value)
(u in R, W in R^{n,m}) : u >= opnorm(W)
note n <= m is enforced WLOG since opnorm(W) = opnorm(W')
W is vectorized column-by-column (i.e. vec(W) in Julia)

barrier from "Interior-Point Polynomial Algorithms in Convex Programming" by Nesterov & Nemirovskii 1994
-logdet(u*I_n - W*W'/u) - log(u)

# TODO don't use ForwardDiff: use identity for inverse of matrix plus I and properties of SVD unitary matrices
# TODO eliminate allocations for incone check
=#

mutable struct EpiNormSpectral <: Cone
    usedual::Bool
    dim::Int
    n::Int
    m::Int
    pnt::AbstractVector{Float64}
    mat::Matrix{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function EpiNormSpectral(n::Int, m::Int, isdual::Bool)
        @assert n <= m
        dim = n*m + 1
        cone = new()
        cone.usedual = isdual
        cone.dim = dim
        cone.n = n
        cone.m = m
        cone.mat = Matrix{Float64}(undef, n, m)
        cone.g = Vector{Float64}(undef, dim)
        cone.H = similar(cone.g, dim, dim)
        cone.H2 = similar(cone.H)
        function barfun(pnt)
            W = reshape(pnt[2:end], n, m)
            u = pnt[1]
            return -logdet(u*I - W*W'/u) - log(u)
        end
        cone.barfun = barfun
        cone.diffres = DiffResults.HessianResult(cone.g)
        return cone
    end
end

EpiNormSpectral(n::Int, m::Int) = EpiNormSpectral(n, m, false)

dimension(cone::EpiNormSpectral) = cone.dim
get_nu(cone::EpiNormSpectral) = cone.n + 1
set_initial_point(arr::AbstractVector{Float64}, cone::EpiNormSpectral) = (@. arr = 0.0; arr[1] = 1.0; arr)
loadpnt!(cone::EpiNormSpectral, pnt::AbstractVector{Float64}) = (cone.pnt = pnt)

function incone(cone::EpiNormSpectral, scal::Float64)
    cone.mat[:] = @view cone.pnt[2:end] # TODO a little slow
    F = svd!(cone.mat) # TODO reduce allocs further
    if F.S[1] >= cone.pnt[1]
        return false
    end

    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.pnt)
    cone.g .= DiffResults.gradient(cone.diffres)
    cone.H .= DiffResults.hessian(cone.diffres)

    return factH(cone)
end
