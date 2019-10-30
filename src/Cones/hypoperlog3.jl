#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

(closure of) hypograph of perspective of natural logarithm (AKA exponential cone)
(u in R, v in R_+, w in R_+) : u <= v*log(w/v)

self-concordant barrier from "A homogeneous interior-point algorithm for nonsymmetric convex conic optimization" by Skajaa & Ye 2014
-log(v*log(w/v) - u) - log(w) - log(v)

TODO
use StaticArrays
=#

mutable struct HypoPerLog3{T <: Real} <: Cone{T}
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

    lwv::T
    vlwvu::T
    g2a::T

    function HypoPerLog3{T}(
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

HypoPerLog3{T}() where {T <: Real} = HypoPerLog3{T}(false)

dimension(cone::HypoPerLog3) = 3

# TODO only allocate the fields we use
function setup_data(cone::HypoPerLog3{T}) where {T <: Real}
    reset_data(cone)
    cone.point = zeros(T, 3)
    cone.grad = zeros(T, 3)
    cone.hess = Symmetric(zeros(T, 3, 3), :U)
    cone.inv_hess = Symmetric(zeros(T, 3, 3), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    return
end

get_nu(cone::HypoPerLog3) = 3

function set_initial_point(arr::AbstractVector, cone::HypoPerLog3)
    arr .= (-0.827838399, 0.805102005, 1.290927713)
    return arr
end

function update_feas(cone::HypoPerLog3)
    @assert !cone.feas_updated
    (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])

    if v > 0 && w > 0
        cone.lwv = log(w / v)
        cone.vlwvu = v * cone.lwv - u
        cone.is_feas = (cone.vlwvu > 0)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::HypoPerLog3)
    @assert cone.is_feas
    (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])

    cone.grad[1] = inv(cone.vlwvu)
    cone.g2a = (1 - cone.lwv) / cone.vlwvu
    cone.grad[2] = cone.g2a - inv(v)
    cone.grad[3] = (-1 - v / cone.vlwvu) / w
    
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoPerLog3)
    @assert cone.grad_updated
    (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])
    H = cone.hess.data

    vwivlwvu = v / cone.vlwvu / w
    H[1, 1] = abs2(cone.grad[1])
    H[1, 2] = cone.g2a / cone.vlwvu
    H[1, 3] = -vwivlwvu / cone.vlwvu
    H[2, 2] = abs2(cone.g2a) + (inv(cone.vlwvu) + inv(v)) / v
    H[2, 3] = -(v * cone.g2a + 1) / cone.vlwvu / w
    H[3, 3] = abs2(vwivlwvu) - cone.grad[3] / w

    cone.hess_updated = true
    return cone.hess
end
