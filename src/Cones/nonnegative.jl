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
    dim::Int
    point::Vector{T}
    dual_point::Vector{T}
    # TODO can remove all of these fields
    prev_scal_point::Vector{T} # for scaling stepper, old scaling applied to an updated primal iterate
    prev_scal_dual_point::Vector{T} # for scaling stepper, old scaling applied to an updated dual iterate
    new_scal_point::Vector{T} # v in MOSEK # TODO this is strong too much currently because smat(v) will always be diagonal
    scaled_point::Vector{T}
    scaling_point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Diagonal{T, Vector{T}}
    inv_hess::Diagonal{T, Vector{T}}
    scaling_updated::Bool # TODO remove

    correction::Vector{T}

    function Nonnegative{T}(
        dim::Int;
        use_scaling::Bool = true,
        use_3order_corr::Bool = true,
        ) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        cone.dim = dim
        cone.use_scaling = use_scaling
        cone.use_3order_corr = use_3order_corr
        return cone
    end
end

use_dual(cone::Nonnegative) = false # self-dual

use_scaling(cone::Nonnegative) = cone.use_scaling # TODO remove from here and just use one in Cones.jl when all cones allow scaling

use_3order_corr(cone::Nonnegative) = cone.use_3order_corr

load_dual_point(cone::Nonnegative, dual_point::AbstractVector) = copyto!(cone.dual_point, dual_point)

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
    cone.grad = similar(cone.point)
    cone.hess = Diagonal(zeros(T, dim))
    cone.inv_hess = Diagonal(zeros(T, dim))
     # TODO initialize at the same time as the initial point
    cone.new_scal_point = similar(cone.point)
    set_initial_point(cone.new_scal_point, cone)
    cone.scaling_point = similar(cone.point)
    set_initial_point(cone.scaling_point, cone)
    cone.correction = zeros(T, dim)
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

function step_max_dist(cone::Nonnegative{T}, s_sol::AbstractVector{T}, z_sol::AbstractVector{T}) where {T <: Real}
    @assert cone.is_feas
    point = cone.point
    dual_point = cone.dual_point
    dist = T(Inf)
    @inbounds for i in eachindex(point)
        if s_sol[i] < 0
            dist = min(dist, -point[i] / s_sol[i])
        end
        if z_sol[i] < 0
            dist = min(dist, -dual_point[i] / z_sol[i])
        end
    end
    return dist
end

function correction(cone::Nonnegative, s_sol::AbstractVector, z_sol::AbstractVector)
    @. cone.correction = s_sol * z_sol / cone.point
    return cone.correction
end

hess_nz_count(cone::Nonnegative, ::Bool) = cone.dim
inv_hess_nz_count(cone::Nonnegative, lower_only::Bool) = hess_nz_count(cone, lower_only)

hess_nz_idxs_col(cone::Nonnegative, j::Int, ::Bool) = j:j
inv_hess_nz_idxs_col(cone::Nonnegative, j::Int, lower_only::Bool) = hess_nz_idxs_col(cone, j, lower_only)
