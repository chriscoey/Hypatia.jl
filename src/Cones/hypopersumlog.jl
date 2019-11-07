#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

(closure of) hypograph of perspective of sum of logarithms
(u in R, v in R_+, w in R_+^n) : u <= v*sum(log.(w/v))

barrier (guessed, reduces to 3-dim exp cone self-concordant barrier)
-log(v*sum(log.(w/v)) - u) - sum(log.(w)) - log(v)
=#

mutable struct HypoPerSumLog{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
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
    vwivlwvu::Vector{T}

    function HypoPerSumLog{T}(
        dim::Int,
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

HypoPerSumLog{T}(dim::Int) where {T <: Real} = HypoPerSumLog{T}(dim, false)

# TODO only allocate the fields we use
function setup_data(cone::HypoPerSumLog{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.vwivlwvu = zeros(T, dim - 2)
    return
end

get_nu(cone::HypoPerSumLog) = cone.dim

function set_initial_point(arr::AbstractVector, cone::HypoPerSumLog)
    (arr[1], arr[2], w) = get_central_ray_hypopersumlog(cone.dim - 2)
    arr[3:end] .= w
    return arr
end

function update_feas(cone::HypoPerSumLog)
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    if v > 0 && all(wi -> wi > 0, w)
        cone.lwv = sum(log(wi / v) for wi in w)
        cone.vlwvu = v * cone.lwv - u
        cone.is_feas = (cone.vlwvu > 0)
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::HypoPerSumLog)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    cone.grad[1] = inv(cone.vlwvu)
    cone.grad[2] = -(cone.lwv - (cone.dim - 2)) / cone.vlwvu - inv(v)
    gden = -1 - inv(cone.lwv - u / v)
    @. cone.grad[3:end] = gden / w
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoPerSumLog)
    @assert cone.grad_updated
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    d = cone.dim - 2
    H = cone.hess.data

    @. cone.vwivlwvu = v / cone.vlwvu / w
    H[1, 1] = abs2(cone.grad[1])
    lvwnivlwvu = (cone.lwv - d) / cone.vlwvu
    H[1, 2] = -lvwnivlwvu / cone.vlwvu
    @. H[1, 3:end] = -cone.vwivlwvu / cone.vlwvu
    H[2, 2] = abs2(lvwnivlwvu) + (d / cone.vlwvu + inv(v)) / v
    hden = (v * lvwnivlwvu - 1) / cone.vlwvu
    @. H[2, 3:end] = hden / w
    @inbounds for j in 1:d
        j2 = 2 + j
        @inbounds for i in 1:j
            H[2 + i, j2] = cone.vwivlwvu[i] * cone.vwivlwvu[j]
        end
        H[j2, j2] -= cone.grad[j2] / w[j]
    end

    cone.hess_updated = true
    return cone.hess
end

# see analysis in https://github.com/lkapelevich/HypatiaBenchmarks.jl/tree/master/centralpoints
function get_central_ray_hypopersumlog(wdim::Int)
    if wdim <= 10
        # lookup points where x = f'(x)
        return central_rays_hypopersumlog[wdim, :]
    end
    # use nonlinear fit for higher dimensions
    if wdim <= 70
        u = -2.647364 / wdim - 0.008411
        v = 0.424679 / wdim + 0.553392
        w = 0.760415 / wdim + 1.001795
    else
        u = -3.016339 / wdim - 0.000078
        v = 0.394332 / wdim + 0.553963
        w = 0.838584 / wdim + 1.000016
    end
    return [u, v, w]
end

const central_rays_hypopersumlog = [
    -0.827838399  0.805102005  1.290927713;
    -0.689609381  0.724604185  1.224619879;
    -0.584372734  0.681280549  1.182421998;
    -0.503500819  0.654485416  1.153054181;
    -0.440285901  0.636444221  1.131466932;
    -0.389979933  0.623569273  1.114979598;
    -0.349256801  0.613977662  1.102014462;
    -0.315769104  0.60658984   1.091577909;
    -0.287837755  0.600745276  1.083013006;
    -0.264242795  0.596018958  1.075868819;
    ]
