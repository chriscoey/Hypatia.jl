#=
Copyright 2018, Chris Coey and contributors

functions and caches for cones
=#

module Cones

using LinearAlgebra
import LinearAlgebra.BlasFloat
import LinearAlgebra.copytri!
import SparseArrays.sparse
import Hypatia.RealOrComplex
import Hypatia.HypCholCache
import Hypatia.hyp_chol!
import Hypatia.hyp_chol_inv!
import Hypatia.HypBKCache
import Hypatia.hyp_bk!
import Hypatia.HypBKSolveCache
import Hypatia.hyp_bk_solve!
import Hypatia.set_min_diag!

abstract type Cone{T <: Real} end

include("orthant.jl")
include("epinorminf.jl")
include("epinormeucl.jl")
include("epipersquare.jl")
include("power.jl")
include("hypoperlog.jl")
include("epiperexp.jl")
include("hypogeomean.jl")
include("epinormspectral.jl")
include("possemideftri.jl")
include("hypoperlogdettri.jl")
include("wsospolyinterp.jl")
# include("wsospolyinterpmat.jl")
# include("wsospolyinterpsoc.jl")

use_dual(cone::Cone) = cone.use_dual
load_point(cone::Cone, point::AbstractVector{T}, scal::T) where {T} = (@. cone.point = point / scal)
load_point(cone::Cone, point::AbstractVector) = copyto!(cone.point, point)
dimension(cone::Cone) = cone.dim

is_feas(cone::Cone) = (cone.feas_updated ? cone.is_feas : update_feas(cone))
grad(cone::Cone) = (cone.grad_updated ? cone.grad : update_grad(cone))
hess(cone::Cone) = (cone.hess_updated ? cone.hess : update_hess(cone))
inv_hess(cone::Cone) = (cone.inv_hess_updated ? cone.inv_hess : update_inv_hess(cone))

# fallbacks

# number of nonzeros in the Hessian and inverse
hess_nz_count(cone::Cone, lower_only::Bool) = (lower_only ? div(cone.dim * (cone.dim + 1), 2) : abs2(cone.dim))
inv_hess_nz_count(cone::Cone, lower_only::Bool) = (lower_only ? div(cone.dim * (cone.dim + 1), 2) : abs2(cone.dim))

# the row indices of nonzero elements in column j
hess_nz_idxs_col(cone::Cone, j::Int, lower_only::Bool) = (lower_only ? (j:cone.dim) : (1:cone.dim))
inv_hess_nz_idxs_col(cone::Cone, j::Int, lower_only::Bool) = (lower_only ? (j:cone.dim) : (1:cone.dim))

reset_data(cone::Cone) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.inv_hess_prod_updated = false)

update_hess_prod(cone::Cone) = nothing

function update_inv_hess_prod(cone::Cone{T}) where {T}
    if !cone.hess_updated
        update_hess(cone)
    end
    copyto!(cone.tmp_hess, cone.hess)
    if isnothing(cone.hess_fact_cache)
        cone.hess_fact_cache = HypBKCache(cone.tmp_hess.uplo, cone.tmp_hess.data)
    end
    cone.hess_fact = hyp_bk!(cone.hess_fact_cache, cone.tmp_hess.data)
    if !issuccess(cone.hess_fact) # TODO maybe better to not step to this point if the hessian factorization fails
        # TODO equilibration makes more sense than current recovery method
        @warn("numerical failure: cannot factorize primitive cone hessian")
        copyto!(cone.tmp_hess, cone.hess)
        set_min_diag!(cone.tmp_hess.data, sqrt(eps(T)))
        cone.hess_fact = hyp_bk!(cone.hess_fact_cache, cone.tmp_hess.data)
        if !issuccess(cone.hess_fact)
            @warn("numerical failure: could not fix failure of positive definiteness")
        end
    end
    cone.inv_hess_prod_updated = true
    return
end

function update_inv_hess(cone::Cone)
    if !cone.inv_hess_prod_updated
        update_inv_hess_prod(cone)
    end
    cone.inv_hess = Symmetric(inv(cone.hess_fact), :U) # TODO use in-place function
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
    return ldiv!(prod, cone.hess_fact, arr) # TODO could use sysvx with already computed factorization here, for improved numerics
end

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
