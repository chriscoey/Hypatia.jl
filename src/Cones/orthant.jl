#=
Copyright 2018, Chris Coey and contributors

nonnegative/nonpositive orthant cones
nonnegative cone: w in R^n : w_i >= 0
nonpositive cone: w in R^n : w_i <= 0

barriers from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
nonnegative cone: -sum_i(log(u_i))
nonpositive cone: -sum_i(log(-u_i))
=#

mutable struct Nonnegative{T <: Real} <: Cone{T}
    use_dual::Bool
    use_scaling::Bool
    dim::Int
    point::Vector{T}
    dual_point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Diagonal{T, Vector{T}}
    inv_hess::Diagonal{T, Vector{T}}

    function Nonnegative{T}(dim::Int, is_dual::Bool) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

Nonnegative{T}(dim::Int) where {T <: Real} = Nonnegative{T}(dim, false)

# mutable struct Nonpositive{T <: Real} <: Cone{T}
#     use_dual::Bool
#     dim::Int
#     point::Vector{T}
#
#     feas_updated::Bool
#     grad_updated::Bool
#     hess_updated::Bool
#     inv_hess_updated::Bool
#     is_feas::Bool
#     grad::Vector{T}
#     hess::Diagonal{T, Vector{T}}
#     inv_hess::Diagonal{T, Vector{T}}
#
#     function Nonpositive{T}(dim::Int, is_dual::Bool, use_scaling::Bool) where {T <: Real}
#         @assert dim >= 1
#         cone = new{T}()
#         cone.use_dual = is_dual
#         cone.use_scaling = use_scaling
#         cone.dim = dim
#         return cone
#     end
# end

# Nonpositive{T}(dim::Int) where {T <: Real} = Nonpositive{T}(dim, false, false)

# const OrthantCone{T <: Real} = Union{Nonnegative{T}, Nonpositive{T}}
const OrthantCone{T <: Real} = Nonnegative{T}

reset_data(cone::OrthantCone) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.WWt_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::OrthantCone{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Diagonal(zeros(T, dim))
    cone.inv_hess = Diagonal(zeros(T, dim))
    return
end

get_nu(cone::OrthantCone) = cone.dim

set_initial_point(arr::AbstractVector, cone::Nonnegative) = (arr .= 1)
# set_initial_point(arr::AbstractVector, cone::Nonpositive) = (arr .= -1)

function update_feas(cone::Nonnegative)
    @assert !cone.feas_updated
    cone.is_feas = all(u -> (u > 0), cone.point)
    cone.feas_updated = true
    return cone.is_feas
end
# function update_feas(cone::Nonpositive)
#     @assert !cone.feas_updated
#     cone.is_feas = all(u -> (u < 0), cone.point)
#     cone.feas_updated = true
#     return cone.is_feas
# end

function update_grad(cone::OrthantCone)
    @assert cone.is_feas
    @. cone.grad = -inv(cone.point)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::OrthantCone)
    @assert cone.grad_updated
    @. cone.hess.diag = abs2(cone.grad)
    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::OrthantCone)
    @assert cone.is_feas
    if cone.use_scaling
        @. cone.WWt.diag = cone.point / cone.dual_point
    else
        @. cone.inv_hess.diag = abs2(cone.point)
    end
    cone.inv_hess_updated = true
    return cone.inv_hess
end

update_hess_prod(cone::OrthantCone) = nothing
update_inv_hess_prod(cone::OrthantCone) = nothing

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::OrthantCone)
    @assert cone.is_feas
    @. prod = arr / cone.point / cone.point
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::OrthantCone)
    @assert cone.is_feas
    @. prod = arr * cone.point * cone.point
    return prod
end

# multiplies arr by W, the squareroot of the scaling matrix
function scalmat_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::OrthantCone)
    @. prod = arr * sqrt(cone.point) / sqrt(cone.dual_point)
    return prod
end

# divides arr by lambda, the scaled point
function scalvec_ldiv!(div, cone::OrthantCone, arr)
    @. div = arr / sqrt(cone.point) / sqrt(cone.dual_point)
    return div
end

# calculates W times lambda inverse times e
function scalmat_scalveci(cone::OrthantCone)
    return inv(cone.dual_point)
end

hess_nz_count(cone::OrthantCone, ::Bool) = cone.dim
inv_hess_nz_count(cone::OrthantCone, lower_only::Bool) = hess_nz_count(cone, lower_only)

hess_nz_idxs_col(cone::OrthantCone, j::Int, ::Bool) = j:j
inv_hess_nz_idxs_col(cone::OrthantCone, j::Int, lower_only::Bool) = hess_nz_idxs_col(cone, j, lower_only)
