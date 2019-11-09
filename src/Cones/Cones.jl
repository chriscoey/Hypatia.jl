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

hessian_cache(T::Type{<:LinearAlgebra.BlasReal}) = DenseSymCache{T}() # use BunchKaufman for BlasReals
hessian_cache(T::Type{<:Real}) = DensePosDefCache{T}() # use Cholesky for generic reals

abstract type Cone{T <: Real} end

include("nonnegative.jl")
include("epinorminf.jl")
include("epinormeucl.jl")
include("epipersquare.jl")
include("power.jl")
include("hypopersumlog.jl")
include("epiperexp.jl")
include("epipersumexp.jl")
include("hypogeomean.jl")
include("epinormspectral.jl")
include("possemideftri.jl")
include("hypoperlogdettri.jl")
include("wsospolyinterp.jl")
# include("wsospolyinterpmat.jl")
# include("wsospolyinterpsoc.jl")

use_scaling(cone::Cone) = false
use_3order_corr(cone::Cone) = false
try_scaled_updates(cone::Cone) = false

use_dual(cone::Cone) = cone.use_dual
load_point(cone::Cone, point::AbstractVector) = copyto!(cone.point, point)
load_dual_point(cone::Cone, dual_point::AbstractVector) = nothing
dimension(cone::Cone) = cone.dim

is_feas(cone::Cone) = (cone.feas_updated ? cone.is_feas : update_feas(cone))
grad(cone::Cone) = (cone.grad_updated ? cone.grad : update_grad(cone))
hess(cone::Cone) = (cone.hess_updated ? cone.hess : update_hess(cone))
inv_hess(cone::Cone) = (cone.inv_hess_updated ? cone.inv_hess : update_inv_hess(cone))
# fallbacks
step_and_update_scaling(::Cone{T}, ::AbstractVector{T}, ::AbstractVector{T}, ::T) where {T} = nothing

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

# utilities for converting between symmetric/Hermitian matrix and vector triangle forms
# TODO fix later, rt2::T doesn't work with tests using ForwardDiff
function smat_to_svec!(vec::AbstractVector{T}, mat::AbstractMatrix{T}, rt2::Number) where {T}
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            vec[k] = mat[i, j]
        else
            vec[k] = mat[i, j] * rt2
        end
        k += 1
    end
    return vec
end

function svec_to_smat!(mat::AbstractMatrix{T}, vec::AbstractVector{T}, rt2::Number) where {T}
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            mat[i, j] = vec[k]
        else
            mat[i, j] = vec[k] / rt2
        end
        k += 1
    end
    return mat
end

function smat_to_svec!(vec::AbstractVector{T}, mat::AbstractMatrix{Complex{T}}, rt2::Number) where {T}
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            vec[k] = real(mat[i, j])
            k += 1
        else
            ck = mat[i, j] * rt2
            vec[k] = real(ck)
            k += 1
            vec[k] = -imag(ck)
            k += 1
        end
    end
    return vec
end

function svec_to_smat!(mat::AbstractMatrix{Complex{T}}, vec::AbstractVector{T}, rt2::Number) where {T}
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            mat[i, j] = vec[k]
            k += 1
        else
            mat[i, j] = Complex(vec[k], -vec[k + 1]) / rt2
            k += 2
        end
    end
    return mat
end


# TODO apply scaling updates per MOSEK and Tuncel papers
# scal_hess(cone::Cone{T}, mu::T) where {T} = (cone.scal_hess_updated ? cone.scal_hess : update_scal_hess(cone, mu))
#
# import Optim
# import ForwardDiff
#
# function update_scal_hess(
#     cone::Cone{T},
#     mu::T;
#     use_update_1::Bool = true,
#     use_update_2::Bool = false,
#     ) where {T}
#     @assert is_feas(cone)
#     @assert !cone.scal_hess_updated
#     s = cone.point
#     z = cone.dual_point
#
#     scal_hess = mu * hess(cone)
#
#     if use_update_1
#         # first update
#         denom_a = dot(s, z)
#         muHs = scal_hess * s
#         denom_b = dot(s, muHs)
#
#         if denom_a > 0
#             scal_hess += Symmetric(z * z') / denom_a
#         end
#         if denom_b > 0
#             scal_hess -= Symmetric(muHs * muHs') / denom_b
#         end
#
#         @show norm(scal_hess * s - z)
#     end
#
#     if use_update_2
#         # second update
#         g = grad(cone)
#         conj_g = conjugate_gradient(cone.barrier, cone.check_feas, s, z)
#
#         mu_cone = dot(s, z) / get_nu(cone)
#         # @show mu_cone
#         dual_gap = z + mu_cone * g
#         primal_gap = s + mu_cone * conj_g
#         # dual_gap = z + mu * g
#         # primal_gap = s + mu * conj_g
#
#         denom_a = dot(primal_gap, dual_gap)
#         H1prgap = scal_hess * primal_gap
#         denom_b = dot(primal_gap, H1prgap)
#
#         if denom_a > 0
#             scal_hess += Symmetric(dual_gap * dual_gap') / denom_a
#         end
#         if denom_b > 0
#             scal_hess -= Symmetric(H1prgap * H1prgap') / denom_b
#         end
#
#         # @show primal_gap, dual_gap
#         @show norm(scal_hess * s - z)
#         @show norm(scal_hess * -conj_g + g)
#         @show norm(scal_hess * primal_gap - dual_gap)
#     end
#
#     copyto!(cone.scal_hess, scal_hess)
#
#     cone.scal_hess_updated = true
#     return cone.scal_hess
# end
#
# # TODO use domain constraints properly
# function conjugate_gradient(barrier::Function, check_feas::Function, s::Vector{T}, z::Vector{T}) where {T}
#     modified_legendre(x) = (check_feas(x) ? dot(z, x) + barrier(x) : Inf)
#     res = Optim.optimize(modified_legendre, s, Optim.Newton())
#     minimizer = Optim.minimizer(res)
#     @assert !any(isnan, minimizer)
#     return -minimizer
# end

end
