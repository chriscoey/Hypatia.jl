#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

hypograph of generalized geomean (product of powers) parametrized by alpha in R_+^n on unit simplex
(u in R, w in R_+^n) : u <= prod_i(w_i^alpha_i)
where sum_i(alpha_i) = 1, alpha_i >= 0

barrier from Constructing self-concordant barriers for convex cones by Yu. Nesterov
-log(prod_i(w_i^alpha_i) - u) - sum_i(log(w_i))
=#

mutable struct HypoGeomean2{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    alpha::Vector{T}
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    wprod::T
    wprodu::T
    tmpn::Vector{T}

    function HypoGeomean2{T}(
        alpha::Vector{T},
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        dim = length(alpha) + 1
        @assert dim >= 2
        @assert all(ai > 0 for ai in alpha)
        tol = 1e3 * eps(T)
        @assert sum(alpha) ≈ 1 atol=tol rtol=tol
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.alpha = alpha
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

HypoGeomean2{T}(alpha::Vector{T}) where {T <: Real} = HypoGeomean2{T}(alpha, false)

function setup_data(cone::HypoGeomean2{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.tmpn = zeros(T, dim - 1)
    return
end

get_nu(cone::HypoGeomean2) = cone.dim

# TODO work out how to get central ray
function set_initial_point(arr::AbstractVector{T}, cone::HypoGeomean2{T}) where {T}
    arr[1] = 0
    @. arr[2:end] = T(2) ^ (1 / cone.alpha)
    return arr
end

function update_feas(cone::HypoGeomean2)
    @assert !cone.feas_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    if all(wi -> wi > 0, w)
        # wprod always calculated because used in update_grad
        cone.wprod = sum(cone.alpha[i] * log(w[i]) for i in eachindex(cone.alpha))
        cone.is_feas = (u < 0) || (cone.wprod > log(u))
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::HypoGeomean2)
    @assert cone.is_feas
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    wprod = cone.wprod = exp(cone.wprod)
    wprodu = cone.wprodu = cone.wprod - u
    cone.grad[1] = inv(wprodu)
    @. cone.tmpn = wprod * cone.alpha / w / wprodu
    @. cone.grad[2:end] = -cone.tmpn - 1 / w
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoGeomean2)
    @assert cone.grad_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    alpha = cone.alpha
    wprod = cone.wprod
    wprodu = cone.wprodu
    wwprodu = wprod / wprodu
    tmpn = cone.tmpn
    H = cone.hess.data

    H[1, 1] = cone.grad[1] / wprodu
    @inbounds for j in eachindex(w)
        j1 = j + 1
        fact = tmpn[j]
        H[1, j1] = -fact / wprodu
        @inbounds for i in 1:(j - 1)
            H[i + 1, j1] = fact * alpha[i] / w[i] * (wwprodu - 1)
        end
        H[j1, j1] = fact * (1 - alpha[j] + alpha[j] * wwprodu) / w[j] + inv(w[j]) / w[j]
    end
    cone.hess_updated = true
    return cone.hess
end

# see analysis in https://github.com/lkapelevich/HypatiaBenchmarks.jl/tree/master/centralpoints
# function get_central_ray_hypogeomean2(alpha::Vector{<:Real})
#     wdim = length(alpha)
#     # predict each w_i given alpha_i and n
#     w = zeros(wdim)
#     if wdim == 1
#         w .= 1.30656
#     elseif wdim == 2
#         @. w = 0.371639 * alpha ^ 3 - 0.408226 * alpha ^ 2 + 0.337555 * alpha + 0.999426
#     elseif wdim <= 5
#         @. w = 0.90687113 - 0.02417035 * log(wdim) + 0.12939174 * exp(alpha)
#     elseif wdim <= 20
#         @. w = 0.927309483 - 0.004331391 * log(wdim) + 0.082597680 * exp(alpha)
#     elseif wdim <= 100
#         @. w = 0.9830810972 - 0.0002152296 * log(wdim) + 0.0177761654 * exp(alpha)
#     else
#         @. w = 9.968391e-01 - 9.605928e-06 * log(wdim) + 3.215512e-03 * exp(alpha)
#     end
#     # get u in closed form from w
#     p = exp(-sum(alpha[i] * log(alpha[i]) for i in eachindex(alpha)))
#     u = sum(p .- alpha .* p ./ (abs2.(w) .- 1)) / wdim
#     return [u, w]
# end
