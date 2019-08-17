#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of matrix spectral norm (operator norm associated with standard Euclidean norm; i.e. maximum singular value)
(u in R, W in R^{n,m}) : u >= opnorm(W)
note n <= m is enforced WLOG since opnorm(W) = opnorm(W')
W is vectorized column-by-column (i.e. vec(W) in Julia)

barrier from "Interior-Point Polynomial Algorithms in Convex Programming" by Nesterov & Nemirovskii 1994
-logdet(u*I_n - W*W'/u) - log(u)

TODO eliminate allocations
TODO type auxiliary fields
=#

mutable struct EpiNormSpectral{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    n::Int
    m::Int
    point::AbstractVector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    W::Matrix{T}
    WWt::Matrix{T}
    Z::Matrix{T}
    fact_Z
    Zi::Symmetric{T, Matrix{T}}
    Eu::Symmetric{T, Matrix{T}}
    ZiEuZi::Matrix{T}
    tmpnn::Matrix{T}
    tmpnm::Matrix{T}

    tmp_hess::Symmetric{T, Matrix{T}}
    hess_fact # TODO prealloc

    function EpiNormSpectral{T}(n::Int, m::Int, is_dual::Bool) where {T <: Real}
        @assert n <= m
        dim = n * m + 1
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.n = n
        cone.m = m
        return cone
    end
end

EpiNormSpectral{T}(n::Int, m::Int) where {T <: Real} = EpiNormSpectral{T}(n, m, false)

function setup_data(cone::EpiNormSpectral{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.W = Matrix{T}(undef, cone.n, cone.m)
    cone.WWt = Matrix{T}(undef, cone.n, cone.n)
    cone.Z = Matrix{T}(undef, cone.n, cone.n)
    cone.Zi = Symmetric(zeros(T, cone.n, cone.n))
    cone.Eu = Symmetric(zeros(T, cone.n, cone.n))
    cone.ZiEuZi = Matrix{T}(undef, cone.n, cone.n)
    cone.tmpnn = Matrix{T}(undef, cone.n, cone.n)
    cone.tmpnm = Matrix{T}(undef, cone.n, cone.m)
    return
end

get_nu(cone::EpiNormSpectral) = cone.n + 1

function set_initial_point(arr::AbstractVector, cone::EpiNormSpectral)
    arr .= 0
    arr[1] = 1
    return arr
end

function update_feas(cone::EpiNormSpectral)
    @assert !cone.feas_updated
    u = cone.point[1]
    if u > 0
        cone.W[:] .= view(cone.point, 2:cone.dim)
        hyp_AAt!(cone.WWt, cone.W)
        cone.Z = u * I - cone.WWt / u
        cone.fact_Z = hyp_chol!(Symmetric(cone.Z, :U))
        cone.is_feas = isposdef(cone.fact_Z)
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiNormSpectral{T}) where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    cone.Zi = Symmetric(inv(cone.fact_Z))
    cone.Eu = I + Symmetric(cone.WWt, :U) / u / u
    cone.grad[1] = -dot(cone.Zi, cone.Eu) - inv(u)
    mul!(cone.tmpnm, cone.Zi, cone.W, T(2) / u, zero(T))
    cone.grad[2:end] .= vec(cone.tmpnm)
    cone.grad_updated = true
    return cone.grad
end

# TODO maybe this could be simpler/faster if use mul!(cone.hess, cone.grad, cone.grad') and build from there
function update_hess(cone::EpiNormSpectral)
    @assert cone.grad_updated
    n = cone.n
    m = cone.m
    u = cone.point[1]
    W = cone.W
    WWt = cone.WWt
    Z = cone.Z
    Zi = cone.Zi
    Eu = cone.Eu
    ZiEuZi = cone.ZiEuZi
    tmpnn = cone.tmpnn
    tmpnm = cone.tmpnm
    cone.hess .= 0
    H = cone.hess.data

    # no BLAS method for product of two symmetric matrices, faster if one is not symmetric
    mul!(tmpnn, Eu, Zi.data)
    mul!(ZiEuZi, Zi, tmpnn)
    H[1, 1] = dot(Symmetric(ZiEuZi, :U), Eu) + (2 * dot(Zi, Symmetric(WWt, :U)) / u + 1) / u / u
    @. tmpnn = -2 * (Zi / u + ZiEuZi) / u
    mul!(tmpnm, tmpnn, W)
    @views copyto!(H[1, 2:end], tmpnm)

    p = 2
    # calculate d^2F / dW_ij dW_kl, p and q are linear indices for (i, j) and (k, l)
    @inbounds for j in 1:m, i in 1:n
        @views mul!(tmpnn, Zi[:, i], W[:, j]')
        mul!(Z, tmpnn, Zi)
         # evaluates to Zi * dZdW_ij * Zi
        tmpnn = (Z + Z') / u
        # add to terms where k = i, and l = j:n, inner product of Zi with d^2Z / dW_ij dW_kl nonzero only when j=l
        q = p
        viewij = view(H, p, q:(q + n - i))
        # inner product of Zi with d^2Z / dW_ij dW_kl, unscaled by 2 / u
        @views viewij .= Zi[i, i:n]
        @views @. for ni in 1:n
            viewij += W[ni, j] * tmpnn[ni, i:n]
        end
        # add to terms where k > i, l = 1:n
        q += (n - i + 1)
        ntermsij = n * (m - j)
        if j <= m - 1
            # the size of the result changes each iteration
            viewH = reshape(view(H, p, q:(q + ntermsij - 1)), n, m - j)
            mul!(viewH, Symmetric(tmpnn, :U), W[:, (j + 1):m], true, true)
        end
        # scale all terms we just modified, i.e. k >= i
        @views H[p, (q - n + i - 1):(q + ntermsij - 1)] .*= 2 / u
        q += ntermsij
        p += 1
    end
    cone.hess_updated = true
    return cone.hess
end
