#=
Copyright 2018, Chris Coey and contributors

epigraph of perspective of (half) square function (AKA rotated second-order cone)
(u in R, v in R_+, w in R^n) : u >= v*1/2*norm_2(w/v)^2
note v*1/2*norm_2(w/v)^2 = 1/2*sum_i(w_i^2)/v

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-log(2*u*v - norm_2(w)^2)
=#

mutable struct EpiPerSquare{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    
    point::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    Hi::Matrix{Float64}

    function EpiPerSquare(dim::Int, is_dual::Bool)
        cone = new()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

EpiPerSquare(dim::Int) = EpiPerSquare(dim, false)

function setup_data(cone::EpiPerSquare)
    dim = cone.dim
    cone.g = Vector{Float64}(undef, dim)
    cone.H = Matrix{Float64}(undef, dim, dim)
    cone.Hi = similar(cone.H)
    return
end

get_nu(cone::EpiPerSquare) = 2

set_initial_point(arr::AbstractVector{Float64}, cone::EpiPerSquare) = (@. arr = 0.0; arr[1] = 1.0; arr[2] = 1.0; arr)

function check_in_cone(cone::EpiPerSquare)
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    if u <= 0.0 || v <= 0.0
        return false
    end
    nrm2 = 0.5*sum(abs2, w)
    dist = u*v - nrm2
    if dist <= 0.0
        return false
    end

    @. cone.g = cone.point / dist
    (cone.g[1], cone.g[2]) = (-cone.g[2], -cone.g[1])

    Hi = cone.Hi
    mul!(Hi, cone.point, cone.point') # TODO syrk
    Hi[2, 1] = Hi[1, 2] = nrm2
    for j in 3:cone.dim
        Hi[j, j] += dist
    end

    H = cone.H
    @. H = Hi
    for j in 3:cone.dim
        H[1, j] = H[j, 1] = -Hi[2, j]
        H[2, j] = H[j, 2] = -Hi[1, j]
    end
    H[1, 1] = Hi[2, 2]
    H[2, 2] = Hi[1, 1]
    @. H *= abs2(inv(dist))

    return true
end

# calcg!(g::AbstractVector{Float64}, cone::EpiPerSquare) = (@. g = cone.point/cone.dist; tmp = g[1]; g[1] = -g[2]; g[2] = -tmp; g)
# calcHiarr!(prod::AbstractVecOrMat{Float64}, arr::AbstractVecOrMat{Float64}, cone::EpiPerSquare) = mul!(prod, cone.Hi, arr)
# calcHarr!(prod::AbstractVecOrMat{Float64}, arr::AbstractVecOrMat{Float64}, cone::EpiPerSquare) = mul!(prod, cone.H, arr)

inv_hess(cone::EpiPerSquare) = Symmetric(cone.Hi, :U)

inv_hess_prod!(prod::AbstractVecOrMat{Float64}, arr::AbstractVecOrMat{Float64}, cone::EpiPerSquare) = mul!(prod, Symmetric(cone.Hi, :U), arr)
