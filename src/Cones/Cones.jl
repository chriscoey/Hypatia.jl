#=
Copyright 2018, Chris Coey and contributors

functions and caches for cones
=#

module Cones

using LinearAlgebra
import LinearAlgebra.copytri!
import GenericLinearAlgebra.eigvals
import GenericLinearAlgebra.svd
import Hypatia.RealOrComplex
import Hypatia.DenseSymCache
import Hypatia.DensePosDefCache
import Hypatia.load_matrix
import Hypatia.update_fact
import Hypatia.solve_system
import Hypatia.invert

import Optim
import ForwardDiff

hessian_cache(T::Type{<:LinearAlgebra.BlasReal}) = DenseSymCache{T}() # use BunchKaufman for BlasReals
hessian_cache(T::Type{<:Real}) = DensePosDefCache{T}() # use Cholesky for generic reals

abstract type Cone{T <: Real} end

include("nonnegative.jl")
include("epinorminf.jl")
include("epinormeucl.jl")
include("epipersquare.jl")
include("power.jl")
include("hypoperlog.jl")
include("epiperexp3.jl")
include("epiperexp.jl")
include("hypogeomean.jl")
include("epinormspectral.jl")
include("possemideftri.jl")
include("hypoperlogdettri.jl")
include("wsospolyinterp.jl")
# include("wsospolyinterpmat.jl")
# include("wsospolyinterpsoc.jl")

use_scaling(cone::Cone) = false
use_3order_corr(cone::Cone) = false
use_dual(cone::Cone) = cone.use_dual
load_point(cone::Cone, point::AbstractVector) = copyto!(cone.point, point)
load_dual_point(cone::Cone, dual_point::AbstractVector) = copyto!(cone.dual_point, dual_point)
dimension(cone::Cone) = cone.dim

is_feas(cone::Cone) = (cone.feas_updated ? cone.is_feas : update_feas(cone))
grad(cone::Cone) = (cone.grad_updated ? cone.grad : update_grad(cone))
hess(cone::Cone) = (cone.hess_updated ? cone.hess : update_hess(cone))
inv_hess(cone::Cone) = (cone.inv_hess_updated ? cone.inv_hess : update_inv_hess(cone))

# fallbacks

# TODO cleanup and make efficient
function get_scaling(cone::Cone{T}, mu::T) where {T}
    s = cone.point
    z = cone.dual_point

    muH = mu * hess(cone)
    dual_gap = cone.dual_point + mu * grad(cone)
    primal_gap = cone.point + mu * conjugate_gradient(cone.barrier, cone.check_feas, cone.dual_point)

    H1 = copy(muH)
    denom = dot(s, z)
    @show denom
    @assert denom >= 0
    if denom > 0
        H1 += z * z' / denom
    end

    denom = dot(s, muH, s)
    @show denom
    @assert denom >= 0
    if denom > 0
        muHs = muH * s
        H1 -= muHs * muHs' / denom
    end

    H2 = copy(H1)
    denom = dot(primal_gap, dual_gap)
    # TODO is it OK if this is negative?
    @show denom
    # @assert denom >= 0
    # if denom > 0
    if !iszero(denom)
        H2 += dual_gap * dual_gap' / denom
    end

    denom = dot(primal_gap, H1, primal_gap)
    @show denom
    @assert denom >= 0
    if denom > 0
        H1prgap = H1 * primal_gap
        H2 -= H1prgap * H1prgap' / denom
    end

    cone.scaling_updated = true
    return H2
end

# number of nonzeros in the Hessian and inverse
function hess_nz_count(cone::Cone, lower_only::Bool)
    dim = dimension(cone)
    if lower_only
        return div(dim * (dim + 1), 2)
    else
        return abs2(dim)
    end
end
inv_hess_nz_count(cone::Cone, lower_only::Bool) = hess_nz_count(cone, lower_only) # NOTE careful: fallback yields same for inv hess as hess

# the row indices of nonzero elements in column j
function hess_nz_idxs_col(cone::Cone, j::Int, lower_only::Bool)
    dim = dimension(cone)
    if lower_only
        return j:dim
    else
        return 1:dim
    end
end
inv_hess_nz_idxs_col(cone::Cone, j::Int, lower_only::Bool) = hess_nz_idxs_col(cone, j, lower_only) # NOTE careful: fallback yields same for inv hess as hess

reset_data(cone::Cone) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.inv_hess_prod_updated = false)

update_hess_prod(cone::Cone) = nothing

function update_inv_hess_prod(cone::Cone{T}) where {T}
    if !cone.hess_updated
        update_hess(cone)
    end
    update_fact(cone.hess_fact_cache, cone.hess)
    # TODO recover if fails - check issuccess
    cone.inv_hess_prod_updated = true
    return
end

function update_inv_hess(cone::Cone)
    if !cone.inv_hess_prod_updated
        update_inv_hess_prod(cone)
    end
    invert(cone.hess_fact_cache, cone.inv_hess)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    if !cone.hess_updated
        update_hess(cone)
    end
    return mul!(prod, cone.hess, arr)
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    if !cone.inv_hess_prod_updated
        update_inv_hess_prod(cone)
    end
    copyto!(prod, arr)
    solve_system(cone.hess_fact_cache, prod)
    return prod
end

# utilities for conjugate barriers
# may need a feas_check and return -Inf
function conjugate_gradient(barrier::Function, check_feas::Function, z::Vector{T}) where {T}
    modified_legendre(x) = (check_feas(x) ? dot(z, x) + barrier(x) : T(Inf))
    bar_grad(x) = ForwardDiff.gradient(modified_legendre, x)
    bar_hess(x) = ForwardDiff.hessian(modified_legendre, x)

    dfc = Optim.TwiceDifferentiableConstraints(fill(-T(Inf), size(z)), fill(T(Inf), size(z)))
    df = Optim.TwiceDifferentiable(modified_legendre, bar_grad, bar_hess, z, inplace = false)
    res = Optim.optimize(df, dfc, z, Optim.IPNewton())
    minimizer = Optim.minimizer(res)

    @assert !any(isnan, minimizer)

    return -minimizer
end

# scalmat_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, mu, cone::Cone) = scalmat_prod!(prod, arr, cone)
# scalmat_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone) = nothing

# utilities for converting between symmetric/Hermitian matrix and vector triangle forms

function mat_U_to_vec_scaled!(vec::AbstractVector{T}, mat::AbstractMatrix{T}) where {T}
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            vec[k] = mat[i, j]
        else
            vec[k] = 2 * mat[i, j]
        end
        k += 1
    end
    return vec
end

function vec_to_mat_U_scaled!(mat::AbstractMatrix{T}, vec::AbstractVector{T}) where {T}
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            mat[i, j] = vec[k]
        else
            mat[i, j] = vec[k] / 2
        end
        k += 1
    end
    return mat
end

function mat_U_to_vec_scaled!(vec::AbstractVector{T}, mat::AbstractMatrix{Complex{T}}) where {T}
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            vec[k] = real(mat[i, j])
            k += 1
        else
            ck = 2 * mat[i, j]
            vec[k] = real(ck)
            k += 1
            vec[k] = -imag(ck)
            k += 1
        end
    end
    return vec
end

function vec_to_mat_U_scaled!(mat::AbstractMatrix{Complex{T}}, vec::AbstractVector{T}) where {T}
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            mat[i, j] = vec[k]
            k += 1
        else
            mat[i, j] = Complex(vec[k], -vec[k + 1]) / 2
            k += 2
        end
    end
    return mat
end

function mat_U_to_vec!(vec::AbstractVector{T}, mat::AbstractMatrix{T}) where {T}
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        vec[k] = mat[i, j]
        k += 1
    end
    return vec
end

function vec_to_mat_U!(mat::AbstractMatrix{T}, vec::AbstractVector{T}) where {T}
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        mat[i, j] = vec[k]
        k += 1
    end
    return mat
end

function mat_U_to_vec!(vec::AbstractVector{T}, mat::AbstractMatrix{Complex{T}}) where {T}
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            vec[k] = real(mat[i, j])
            k += 1
        else
            ck = mat[i, j]
            vec[k] = real(ck)
            k += 1
            vec[k] = -imag(ck)
            k += 1
        end
    end
    return vec
end

function vec_to_mat_U!(mat::AbstractMatrix{Complex{T}}, vec::AbstractVector{T}) where {T}
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            mat[i, j] = vec[k]
            k += 1
        else
            mat[i, j] = Complex(vec[k], -vec[k + 1])
            k += 2
        end
    end
    return mat
end

end
