#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

generalized power cone parametrized by alpha in R_++^n in unit simplex interior
(u in R_++^m, w in R^n) : prod_i(u_i^alpha_i) => norm_2(w)
where sum_i(alpha_i) = 1, alpha_i > 0

barrier from "On self-concordant barriers for generalized power cones" by Roy & Xiao 2018
-log(prod_i((u_i)^(2 * alpha_i)) - norm_2(w)^2) - sum_i((1 - alpha_i)*log(u_i))
=#

mutable struct Power{T <: Real} <: Cone{T}
    use_dual::Bool
    max_neighborhood::T
    use_heuristic_neighborhood::Bool
    dim::Int
    alpha::Vector{T}
    n::Int
    point::Vector{T}
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    produ::T
    produw::T
    produuw::T
    aui::Vector{T}
    auiproduuw::Vector{T}

    function Power{T}(
        alpha::Vector{T},
        n::Int,
        is_dual::Bool;
        max_neighborhood::Real = default_max_neighborhood(),
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert n >= 1
        dim = length(alpha) + n
        @assert dim >= 3
        @assert all(ai > 0 for ai in alpha)
        @assert sum(alpha) ≈ 1
        cone = new{T}()
        cone.n = n
        cone.use_dual = is_dual
        cone.max_neighborhood = max_neighborhood
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.dim = dim
        cone.alpha = alpha
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

Power{T}(alpha::Vector{T}, n::Int) where {T <: Real} = Power{T}(alpha, n, false)

dimension(cone::Power) = length(cone.alpha) + cone.n

# TODO only allocate the fields we use
function setup_data(cone::Power{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.aui = zeros(length(cone.alpha))
    cone.auiproduuw = zeros(length(cone.alpha))
    return
end

get_nu(cone::Power) = length(cone.alpha) + 1

function set_initial_point(arr::AbstractVector, cone::Power)
    m = length(cone.alpha)
    @. arr[1:m] = sqrt(1 + cone.alpha)
    arr[(m + 1):cone.dim] .= 0
    return arr
end

function update_feas(cone::Power{T}) where {T <: Real}
    @assert !cone.feas_updated
    m = length(cone.alpha)
    u = cone.point[1:m]
    w = view(cone.point, (m + 1):cone.dim)

    if all(>(zero(T)), u)
        cone.produ = exp(2 * sum(cone.alpha[i] * log(u[i]) for i in eachindex(cone.alpha)))
        cone.produw = cone.produ - sum(abs2, w)
        cone.is_feas = (cone.produw > 0)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::Power)
    @assert cone.is_feas
    m = length(cone.alpha)
    u = cone.point[1:m]
    w = view(cone.point, (m + 1):cone.dim)

    @. cone.aui = 2 * cone.alpha / u
    cone.produuw = cone.produ / cone.produw
    @. cone.auiproduuw = -cone.aui * cone.produuw
    @. cone.grad[1:m] = cone.auiproduuw - (1 - cone.alpha) / u
    produwi2 = 2 / cone.produw
    @. cone.grad[(m + 1):end] = produwi2 * w

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::Power)
    @assert cone.grad_updated
    m = length(cone.alpha)
    u = cone.point[1:m]
    w = view(cone.point, (m + 1):cone.dim)
    aui = cone.aui
    auiproduuw = cone.auiproduuw
    g = cone.grad
    H = cone.hess.data

    produuwm1 = 1 - cone.produuw
    @inbounds for j in 1:m
        auiproduuwm1 = auiproduuw[j] * produuwm1
        @inbounds for i in 1:j
            H[i, j] = aui[i] * auiproduuwm1
        end
        H[j, j] -= g[j] / u[j]
    end

    offset = 2 / cone.produw
    for j in m .+ (1:cone.n)
        gj = g[j]
        @inbounds for i in 1:m
            H[i, j] = auiproduuw[i] * gj
        end
        @inbounds for i in (m + 1):j
            H[i, j] = g[i] * gj
        end
        H[j, j] += offset
    end

    cone.hess_updated = true
    return cone.hess
end

# TODO update and benchmark to decide whether this improves speed/numerics
# function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Power)
#     @assert cone.grad_updated
#     m = length(cone.alpha)
#     dim = cone.dim
#     u = cone.point[1:m]
#     w = view(cone.point, (m + 1):dim)
#     alpha = cone.alpha
#     produw = cone.produw
#     tmpm = cone.tmpm
#     aui = cone.aui
#     produuw = cone.produuw
#     @. tmpm = 2 * produuw * aui / produw
#
#     @. @views prod[1:m, :] = aui * produuw * (produuw - 1)
#     @. @views prod[(m + 1):dim, :] = 2 / produw
#     @views @inbounds for i in 1:size(arr, 2)
#         dotm = dot(aui, arr[1:m, i])
#         dotn = dot(w, arr[(m + 1):dim, i])
#         prod[1:m, i] .*= dotm
#         prod[(m + 1):dim, i] .*= dotn
#         @. prod[1:m, i] -= tmpm * dotn
#         @. prod[(m + 1):dim, i] -= produuw * dotm
#     end
#     @. @views begin
#         prod[1:m, :] += arr[1:m, :] * (produuw * aui + (1 - alpha) / u) / u
#         prod[(m + 1):dim, :] *= w
#         prod[(m + 1):dim, :] += arr[(m + 1):dim, :]
#         prod[(m + 1):dim, :] *= 2
#         prod[(m + 1):dim, :] /= produw
#     end
#
#     return prod
# end
