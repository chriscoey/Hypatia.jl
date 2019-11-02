#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

nonnegative orthant cone
w in R^n : w_i >= 0

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-sum_i(log(u_i))
=#

mutable struct Nonnegative{T <: Real} <: Cone{T}
    use_scaling::Bool
    use_3order_corr::Bool
    scaled_updates::Bool
    scaling_initialized::Bool
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
    scaling_updated::Bool

    scaling_point::Vector{T}
    correction::Vector{T}

    function Nonnegative{T}(
        dim::Int;
        use_scaling::Bool = true,
        use_3order_corr::Bool = true,
        scaled_updates::Bool = true,
        ) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        cone.dim = dim
        cone.use_scaling = use_scaling
        cone.use_3order_corr = use_3order_corr
        cone.scaled_updates = scaled_updates
        cone.scaling_initialized = false
        return cone
    end
end

use_dual(cone::Nonnegative) = false # self-dual

use_scaling(cone::Nonnegative) = cone.use_scaling # TODO remove from here and just use one in Cones.jl when all cones allow scaling

use_3order_corr(cone::Nonnegative) = cone.use_3order_corr

load_dual_point(cone::Nonnegative, dual_point::AbstractVector) = copyto!(cone.dual_point, dual_point)

reset_data(cone::Nonnegative) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = scaling_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::Nonnegative{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Diagonal(zeros(T, dim))
    cone.inv_hess = Diagonal(zeros(T, dim))
    cone.scaling_point = ones(T, dim)
    cone.correction = zeros(T, dim)
    if cone.scaled_updates
    end
    return
end

get_nu(cone::Nonnegative) = cone.dim

set_initial_point(arr::AbstractVector, cone::Nonnegative) = (arr .= 1)

function update_feas(cone::Nonnegative)
    @assert !cone.feas_updated
    cone.is_feas = all(u -> (u > 0), cone.point)
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::Nonnegative)
    @assert cone.is_feas
    @. cone.grad = -inv(cone.point)
    cone.grad_updated = true
    return cone.grad
end

function update_scaling(cone::Nonnegative)
    if cone.scaling_initialized && cone.scaled_updates
        # s, z under the old scaling, in the future thses may already be calculated outside of this oracle
        point2 = scalmat_ldiv!(similar(cone.point), cone.point, cone)
        dual_point2 = scalmat_prod!(similar(cone.point), cone.dual_point, cone)
        @. cone.scaling_point *= sqrt(point2) / sqrt(dual_point2)
        cone.scaling_initialized = true
    else
        @. cone.scaling_point = sqrt(cone.point) / sqrt(cone.dual_point)
    end
    cone.scaling_updated = true
    return cone.scaling_updated
end

function update_hess(cone::Nonnegative)
    if cone.use_scaling
        @assert cone.is_feas
        @. cone.hess.diag = cone.dual_point / cone.point
    else
        @assert cone.grad_updated
        @. cone.hess.diag = abs2(cone.grad)
    end
    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::Nonnegative)
    @assert cone.is_feas
    if cone.use_scaling
        @. cone.inv_hess.diag = cone.point / cone.dual_point
    else
        @. cone.inv_hess.diag = abs2(cone.point)
    end
    cone.inv_hess_updated = true
    return cone.inv_hess
end

update_hess_prod(cone::Nonnegative) = nothing
update_inv_hess_prod(cone::Nonnegative) = nothing

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Nonnegative)
    @assert cone.is_feas
    if cone.use_scaling
        @. prod = arr * cone.dual_point / cone.point
    else
        @. prod = arr / cone.point / cone.point
    end
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Nonnegative)
    @assert cone.is_feas
    if cone.use_scaling
        @. prod = arr * cone.point / cone.dual_point
    else
        @. prod = arr * cone.point * cone.point
    end
    return prod
end

# multiplies arr by W, the squareroot of the scaling matrix
function scalmat_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Nonnegative)
    @. prod = arr * cone.scaling_point
    return prod
end

# scaling is symmetric, trans kwarg ignored TODO factor as another function?
function scalmat_ldiv!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Nonnegative; trans::Bool = false)
    @. prod = arr / cone.scaling_point
    return prod
end

# divides arr by lambda, the scaled point
# TODO think better about whether this oracle is needed
function scalvec_ldiv!(div::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Nonnegative)
    @. div = arr / sqrt(cone.point * cone.dual_point)
    return div
end

function dist_to_bndry(cone::Nonnegative{T}, point::Vector{T}, dir::AbstractVector{T}) where {T}
    dist = T(Inf)
    @inbounds for i in eachindex(point)
        if dir[i] < 0
            dist = min(dist, -point[i] / dir[i])
        end
    end
    return dist
end

# TODO optimize this
function step_max_dist(cone::Nonnegative{T}, s_sol::AbstractVector{T}, z_sol::AbstractVector{T}) where {T}
    @assert cone.is_feas

    # TODO this could go in Cones.jl
    primal_dist = dist_to_bndry(cone, cone.point, s_sol)
    dual_dist = dist_to_bndry(cone, cone.dual_point, z_sol)
    step_dist = min(primal_dist, dual_dist)

    return step_dist
end

# returns lambda_inv * W_inv * correction = grad * correction
function correction(cone::Nonnegative, s_sol::AbstractVector, z_sol::AbstractVector)
    @. cone.correction = s_sol * z_sol / cone.point
    return cone.correction
end

function conic_prod!(w::AbstractVector, u::AbstractVector, v::AbstractVector, cone::Nonnegative)
    @. w = u * v
    return w
end

hess_nz_count(cone::Nonnegative, ::Bool) = cone.dim
inv_hess_nz_count(cone::Nonnegative, lower_only::Bool) = hess_nz_count(cone, lower_only)

hess_nz_idxs_col(cone::Nonnegative, j::Int, ::Bool) = j:j
inv_hess_nz_idxs_col(cone::Nonnegative, j::Int, lower_only::Bool) = hess_nz_idxs_col(cone, j, lower_only)
