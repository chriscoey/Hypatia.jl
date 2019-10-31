#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

(closure of) epigraph of perspective of exponential (AKA exponential cone)
(u in R_+, v in R_+, w in R) : u >= v*exp(w/v)

self-concordant barrier from "A homogeneous interior-point algorithm for nonsymmetric convex conic optimization" by Skajaa & Ye 2014
-log(v*log(u/v) - w) - log(u) - log(v)

TODO
use StaticArrays
=#

mutable struct EpiPerExp3{T <: Real} <: Cone{T}
    use_scaling::Bool
    use_dual::Bool
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

    luv::T
    vluvw::T
    g1a::T
    g2a::T
    correction::Vector{T}

    function EpiPerExp3{T}(
        is_dual::Bool;
        use_scaling::Bool = false, # TODO MOSEK paper scaling
        hess_fact_cache = hessian_cache(T), # TODO delete
        ) where {T <: Real}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

EpiPerExp3{T}() where {T <: Real} = EpiPerExp3{T}(false)

dimension(cone::EpiPerExp3) = 3

use_scaling(cone::EpiPerExp3) = cone.use_scaling # TODO remove from here and just use one in Cones.jl when all cones allow scaling

# TODO only allocate the fields we use
function setup_data(cone::EpiPerExp3{T}) where {T <: Real}
    reset_data(cone)
    cone.point = zeros(T, 3)
    cone.grad = zeros(T, 3)
    cone.hess = Symmetric(zeros(T, 3, 3), :U)
    cone.inv_hess = Symmetric(zeros(T, 3, 3), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.correction = zeros(T, 3)
    return
end

get_nu(cone::EpiPerExp3) = 3

function set_initial_point(arr::AbstractVector, cone::EpiPerExp3)
    arr .= (1.290928, 0.805102, -0.827838) # from MOSEK paper
    return arr
end

function update_feas(cone::EpiPerExp3)
    @assert !cone.feas_updated
    (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])

    if u > 0 && v > 0
        cone.luv = log(u / v)
        cone.vluvw = v * cone.luv - w
        cone.is_feas = (cone.vluvw > 0)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiPerExp3)
    @assert cone.is_feas
    (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])
    vluvw = cone.vluvw

    cone.g1a = v / u / vluvw
    cone.grad[1] = -inv(u) - cone.g1a
    cone.g2a = (1 - cone.luv) / vluvw
    cone.grad[2] = cone.g2a - inv(v)
    cone.grad[3] = inv(vluvw)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiPerExp3)
    @assert cone.grad_updated
    (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])
    H = cone.hess.data
    vluvw = cone.vluvw
    g1a = cone.g1a
    g2a = cone.g2a

    H[1, 3] = -g1a / vluvw
    H[2, 3] = g2a / vluvw
    H[3, 3] = abs2(cone.grad[3])
    H[1, 1] = abs2(g1a) - cone.grad[1] / u
    H[1, 2] = -(v * cone.g2a + 1) / cone.vluvw / u
    H[2, 2] = abs2(g2a) + (inv(vluvw) + inv(v)) / v

    # TODO would we use this?
    # find an R factor for F''(point) = R(x) R(x)'
    # R = similar(H)
    # R[1,1] = (1 - sqrt(1 + 2v)) / vluvw
    # etc

    cone.hess_updated = true
    return cone.hess
end

function correction(cone::EpiPerExp3, s_sol::AbstractVector, z_sol::AbstractVector)
    # TODO F'''
    cone.correction ./= 2
    return cone.correction
end
