#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of Euclidean (2-)norm (AKA second-order cone)
(u in R, w in R^n) : u >= norm_2(w)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd, halved
-log(u^2 - norm_2(w)^2) / 2
=#

mutable struct EpiNormEucl{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    dist::T

    function EpiNormEucl{T}(dim::Int, is_dual::Bool) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

EpiNormEucl{T}(dim::Int) where {T <: Real} = EpiNormEucl{T}(dim, false)

reset_data(cone::EpiNormEucl) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::EpiNormEucl{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    return
end

get_nu(cone::EpiNormEucl) = 1

function set_initial_point(arr::AbstractVector, cone::EpiNormEucl{T}) where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

function update_feas(cone::EpiNormEucl)
    @assert !cone.feas_updated
    u = cone.point[1]
    if u > 0
        w = view(cone.point, 2:cone.dim)
        cone.dist = (abs2(u) - sum(abs2, w))
        cone.is_feas = (cone.dist > 0)
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiNormEucl)
    @assert cone.is_feas
    @. cone.grad = cone.point / cone.dist
    cone.grad[1] *= -1
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiNormEucl)
    @assert cone.grad_updated
    mul!(cone.hess.data, 2 * cone.grad, cone.grad')
    cone.hess += inv(cone.dist) * I
    cone.hess[1, 1] -= 2 / cone.dist
    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::EpiNormEucl)
    @assert cone.is_feas
    mul!(cone.inv_hess.data, 2 * cone.point, cone.point')
    cone.inv_hess += cone.dist * I
    cone.inv_hess[1, 1] -= 2 * cone.dist
    cone.inv_hess_updated = true
    return cone.inv_hess
end

update_hess_prod(cone::EpiNormEucl) = nothing
update_inv_hess_prod(cone::EpiNormEucl) = nothing

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas
    p1 = cone.point[1]
    p2 = @view cone.point[2:end]
    @inbounds for j in 1:size(prod, 2)
        arr_1j = arr[1, j]
        arr_2j = @view arr[2:end, j]
        ga = dot(p2, arr_2j) - p1 * arr_1j
        ga /= (cone.dist / 2)
        prod[1, j] = -ga * p1 - arr_1j
        @. prod[2:end, j] = ga * p2 + arr_2j
    end
    @. prod ./= cone.dist
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas
    @inbounds for j in 1:size(prod, 2)
        @views pa = dot(cone.point, arr[:, j])
        @. prod[:, j] = pa * cone.point * 2
    end
    @. @views prod[1, :] -= cone.dist * arr[1, :]
    @. @views prod[2:end, :] += cone.dist * arr[2:end, :]
    return prod
end


function foo()
    dual_dist = abs2(cone.dual_point[1]) - sum(abs2, cone.dual_point[2:end])
    scaled_primal = cone.point ./ sqrt(cone.dist)
    scaled_dual = cone.dual_point ./ sqrt(dual_dist)

    gamma = sqrt((1 + dot(scaled_primal, scaled_dual)) / 2) # TODO maybe not /2

    w[1] = scaled_primal[1] + scaled_dual[1]
    w[2:end] = (scaled_primal[2:end] - scaled_dual[2:end]) / 2 / gamma # TODO probably not /2

    W = w * w'♠
    W[2:end, 2:end] ./= (w[1] + 1)
    W[2:end, 2:end] += I
    W .*= sqrt(sqrt(primal_dist / dual_dist))

    lambda_scaled[1] = gamma
    lambda_scaled[2:end] = (gamma + scaled_dual[1]) * scaled_primal[2:end] + (gamma + scaled_primal[1]) * scaled_dual[2:end]
    lambda_scaled[2:end] /= (scaled_primal[1] + scaled_dual[1] + 2 * gamma)

    lambda = sqrt(sqrt(primal_dist * primal_dist)) * lambda_scaled

end

# # multiplies arr by W, the squareroot of the scaling matrix
# function scalmat_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::OrthantCone)
#     @. prod = arr * sqrt(cone.point / cone.dual_point)
#     return prod
# end
