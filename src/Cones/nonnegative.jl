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
    try_scaled_updates::Bool # experimental, run algorithm in scaled variables for numerical reasons. it may be too tricky to keep this boolean.
    dim::Int
    point::Vector{T}
    dual_point::Vector{T}
    prev_scal_point::Vector{T} # for scaling stepper, old scaling applied to an updated primal iterate
    prev_scal_dual_point::Vector{T} # for scaling stepper, old scaling applied to an updated dual iterate
    new_scal_point::Vector{T} # v in MOSEK # TODO this is strong too much currently because smat(v) will always be diagonal

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Diagonal{T, Vector{T}}
    inv_hess::Diagonal{T, Vector{T}}
    scaling_updated::Bool # TODO not currently checked or relied on

    correction::Vector{T}

    function Nonnegative{T}(
        dim::Int;
        use_scaling::Bool = true,
        use_3order_corr::Bool = true,
        try_scaled_updates::Bool = true,
        ) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        cone.dim = dim
        cone.use_scaling = use_scaling
        cone.use_3order_corr = use_3order_corr
        cone.try_scaled_updates = try_scaled_updates
        return cone
    end
end

use_dual(cone::Nonnegative) = false # self-dual

use_scaling(cone::Nonnegative) = cone.use_scaling # TODO remove from here and just use one in Cones.jl when all cones allow scaling

use_3order_corr(cone::Nonnegative) = cone.use_3order_corr

# TODO this could replace load_point
load_scaled_point(cone::Nonnegative, point::AbstractVector) = copyto!(cone.scaled_point, point)

reset_data(cone::Nonnegative) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = scaling_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::Nonnegative{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = similar(cone.point)
    cone.prev_scal_point = similar(cone.point)
    cone.prev_scal_dual_point = similar(cone.point)
    cone.new_scal_point = similar(cone.point)
    cone.grad = similar(cone.point)
    cone.hess = Diagonal(zeros(T, dim))
    cone.inv_hess = Diagonal(zeros(T, dim))
     # TODO initialize at the same time as the initial point
    cone.scaled_point = similar(cone.point)
    set_initial_point(cone.scaled_point, cone)
    cone.scaling_point = similar(cone.point)
    set_initial_point(cone.scaling_point, cone)
    cone.correction = zeros(T, dim)
    if cone.try_scaled_updates
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

# calculates the gradient at the true, unscaled primal point
function update_grad(cone::Nonnegative)
    @assert cone.is_feas
    @. cone.grad = -inv(cone.point)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::Nonnegative)
    if cone.use_scaling
        @assert cone.is_feas
        if cone.try_scaled_updates
            @. cone.hess.diag = inv.(abs2.(cone.scaling_point))
        else
            @. cone.hess.diag = cone.dual_point / cone.point
        end
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
        if cone.try_scaled_updates
            @. cone.inv_hess.diag = abs2.(cone.scaling_point)
        else
            @. cone.inv_hess.diag = cone.point / cone.dual_point
        end
    else
        @. cone.inv_hess.diag = abs2(cone.point)
    end
    cone.inv_hess_updated = true
    return cone.inv_hess
end

update_hess_prod(cone::Nonnegative) = nothing
update_inv_hess_prod(cone::Nonnegative) = nothing

scal_hess(cone::Nonnegative{T}, mu::T) where {T} = hess(cone)

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Nonnegative)
    @assert cone.is_feas
    if cone.use_scaling
        if cone.try_scaled_updates
            @. prod = arr / abs2.(cone.scaling_point)
        else
            @. prod = arr * cone.dual_point / cone.point
        end
    else
        @. prod = arr / cone.point / cone.point
    end
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Nonnegative)
    @assert cone.is_feas
    if cone.use_scaling
        if cone.try_scaled_updates
            @. prod = arr * abs2.(cone.scaling_point)
        else
            @. prod = arr * cone.point / cone.dual_point
        end
    else
        @. prod = arr * cone.point * cone.point
    end
    return prod
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
# TODO this could go in Cones.jl
# scales directions, which are stored in cone.s_dir and cone.z_dir and used later
function step_max_dist(cone::Nonnegative, s_sol::AbstractVector, z_sol::AbstractVector)
    @assert cone.is_feas
    # if cone.try_scaled_updates
    #     @. cone.s_dir = s_sol / cone.scaling_point
    #     @. cone.z_dir = z_sol * cone.scaling_point
    #     primal_dist = dist_to_bndry(cone, cone.scaled_point, cone.s_dir)
    #     dual_dist = dist_to_bndry(cone, cone.scaled_point, cone.z_dir)
    # else
        @. cone.s_dir = s_sol
        @. cone.z_dir = z_sol
        primal_dist = dist_to_bndry(cone, cone.point, s_sol)
        dual_dist = dist_to_bndry(cone, cone.dual_point, z_sol)
    # end
    step_dist = min(primal_dist, dual_dist)

    return step_dist
end

# returns scaled_point \ W_inv * correction = grad * correction
function correction(cone::Nonnegative, s_sol::AbstractVector, z_sol::AbstractVector, primal_point)
    if cone.try_scaled_updates
        # TODO pass unscaled point into this oracle, can see differences in number of iters here
        @. cone.correction = s_sol * z_sol / primal_point
    else
        @. cone.correction = s_sol * z_sol / cone.point
    end
    return cone.correction
end

function update_scaling(cone::Nonnegative)
    if cone.try_scaled_updates
        @. cone.scaling_point *= sqrt(cone.point) / sqrt(cone.dual_point)
        @. cone.scaled_point = sqrt(cone.point) .* sqrt(cone.dual_point)
    else
        @. cone.scaling_point = sqrt(cone.point) / sqrt(cone.dual_point)
    end
    cone.scaling_updated = true
    return cone.scaling_updated
end

# this is an oracle for now because we could get the next s, z but in the old scaling by dividing by sqrt(H(v)), which is cone-specific
function step(cone::Nonnegative{T}, step_size::T) where {T}
    # get the next s, z but in the old scaling
    if cone.try_scaled_updates
        # s_next = cone.point
        # z_next = cone.dual_point
        # copyto!(s_next, s_sol)
        # copyto!(z_next, z_sol)
        # @. s_next *= step_size
        # @. z_next *= step_size
        # @. s_next += one(T)
        # @. z_next += one(T)
        # @. s_next *= cone.scaled_point
        # @. z_next *= cone.scaled_point
        @. cone.point = cone.scaled_point + step_size * s_sol / cone.scaling_point
        @. cone.dual_point = cone.scaled_point + step_size * z_sol * cone.scaling_point
    else
        @. cone.point += step_size * s_sol
        @. cone.dual_point += step_size * z_sol
    end
    return
end

# s_sol and z_sol are scaled by an old scaling
function step_and_update_scaling(cone::Nonnegative{T}, step_size::T) where {T}
    step(cone, step_size)
    update_scaling(cone)
    return
end

hess_nz_count(cone::Nonnegative, ::Bool) = cone.dim
inv_hess_nz_count(cone::Nonnegative, lower_only::Bool) = hess_nz_count(cone, lower_only)

hess_nz_idxs_col(cone::Nonnegative, j::Int, ::Bool) = j:j
inv_hess_nz_idxs_col(cone::Nonnegative, j::Int, lower_only::Bool) = hess_nz_idxs_col(cone, j, lower_only)
