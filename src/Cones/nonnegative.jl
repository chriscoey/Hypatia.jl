#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

nonnegative orthant cone:
w in R^n : w_i >= 0

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-sum_i(log(u_i))
=#

mutable struct Nonnegative{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_3order_corr::Bool
    max_neighborhood::T
    dim::Int
    point::Vector{T}
    dual_point::Vector{T}
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Diagonal{T, Vector{T}}
    inv_hess::Diagonal{T, Vector{T}}
    correction::Vector{T}

    function Nonnegative{T}(
        dim::Int;
        use_dual::Bool = false, # TODO self-dual so maybe remove this option/field?
        max_neighborhood::Real = default_max_neighborhood(),
        ) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.use_3order_corr = false # TODO maybe make it a function rather than a field
        cone.max_neighborhood = max_neighborhood
        cone.dim = dim
        return cone
    end
end

use_heuristic_neighborhood(cone::Nonnegative) = false

use_3order_corr(cone::Nonnegative) = cone.use_3order_corr

reset_data(cone::Nonnegative) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

use_sqrt_oracles(cone::Nonnegative) = true

# TODO only allocate the fields we use
function setup_data(cone::Nonnegative{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Diagonal(zeros(T, dim))
    cone.inv_hess = Diagonal(zeros(T, dim))
    cone.correction = zeros(T, dim)
    return
end

get_nu(cone::Nonnegative) = cone.dim

set_initial_point(arr::AbstractVector, cone::Nonnegative) = (arr .= 1)

function update_feas(cone::Nonnegative{T}) where {T}
    @assert !cone.feas_updated
    cone.is_feas = all(>(eps(T)), cone.point)
    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(cone::Nonnegative{T}) where {T} = all(>(eps(T)), cone.dual_point)

function update_grad(cone::Nonnegative)
    @assert cone.is_feas
    @. cone.grad = -inv(cone.point)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::Nonnegative)
    if !cone.grad_updated
        update_grad(cone)
    end
    @. cone.hess.diag = abs2(cone.grad)
    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::Nonnegative)
    @assert cone.is_feas
    @. cone.inv_hess.diag = abs2(cone.point)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Nonnegative)
    @assert cone.is_feas
    @. prod = arr / cone.point / cone.point
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Nonnegative)
    @assert cone.is_feas
    @. prod = arr * cone.point * cone.point
    return prod
end

function hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Nonnegative)
    @assert cone.is_feas
    @. prod = arr / cone.point
    return prod
end

function inv_hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Nonnegative)
    @assert cone.is_feas
    @. prod = arr * cone.point
    return prod
end

hess_nz_count(cone::Nonnegative) = cone.dim
hess_nz_count_tril(cone::Nonnegative) = cone.dim
inv_hess_nz_count(cone::Nonnegative) = cone.dim
inv_hess_nz_count_tril(cone::Nonnegative) = cone.dim
hess_nz_idxs_col(cone::Nonnegative, j::Int) = [j]
hess_nz_idxs_col_tril(cone::Nonnegative, j::Int) = [j]
inv_hess_nz_idxs_col(cone::Nonnegative, j::Int) = [j]
inv_hess_nz_idxs_col_tril(cone::Nonnegative, j::Int) = [j]

function in_neighborhood(cone::Nonnegative{T}, rtmu::T, max_nbhd::T) where {T <: Real}
    mu_nbhd = rtmu * max_nbhd
    return all(abs(si * zi - rtmu) < mu_nbhd for (si, zi) in zip(cone.point, cone.dual_point))
end

# function correction(cone::Nonnegative, primal_dir::AbstractVector, dual_dir::AbstractVector)
#     @. cone.correction = primal_dir * dual_dir / cone.point
#     return cone.correction
# end

function correction(cone::Nonnegative, primal_dir::AbstractVector)
    @. cone.correction = (primal_dir / cone.point)^2 / cone.point
    return cone.correction
end
