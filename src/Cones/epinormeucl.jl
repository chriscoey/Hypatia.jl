#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of Euclidean (2-)norm (AKA second-order cone)
(u in R, w in R^n) : u >= norm_2(w)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd, halved
-log(u^2 - norm_2(w)^2) / 2

TODO tidy up modifications due to halving barrier
=#

mutable struct EpiNormEucl{T <: Real} <: Cone{T}
    use_dual::Bool
    use_scaling::Bool
    dim::Int
    point::Vector{T}
    dual_point::Vector{T}
    scaled_point::Vector{T}
    scaled_dual_point::Vector{T}
    w::Vector{T} #think about naming, it's w_bar in cvxopt paper

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    scaling_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    dist::T
    dual_dist::T
    gamma::T

    mehrotra_correction::Vector{T}

    function EpiNormEucl{T}(dim::Int, is_dual::Bool) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        use_scaling = true # TODO make an option
        return cone
    end
end

EpiNormEucl{T}(dim::Int) where {T <: Real} = EpiNormEucl{T}(dim, false)

reset_data(cone::EpiNormEucl) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.scaling_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::EpiNormEucl{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)

    cone.dual_point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.scaled_point = zeros(T, dim)
    cone.scaled_dual_point = zeros(T, dim)
    cone.w = zeros(T, dim)

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

function update_scaling(cone::EpiNormEucl)
    @assert !cone.scaling_updated
    @assert cone.feas_updated
    dual_dist = cone.dual_dist = abs2(cone.dual_point[1]) - sum(abs2, cone.dual_point[2:end])
    @assert dual_dist >= 0
    scaled_point = cone.scaled_point = cone.point ./ sqrt(cone.dist)
    scaled_dual_point = cone.scaled_dual_point = cone.dual_point ./ sqrt(dual_dist)
    gamma = cone.gamma = sqrt((1 + dot(scaled_point, scaled_dual_point)) / 2)

    cone.w[1] = scaled_point[1] + scaled_dual_point[1]
    cone.w[2:end] = (scaled_point[2:end] - scaled_dual_point[2:end])
    cone.w ./= 2 # NOTE probably not /2 for unhalved barrier
    cone.w ./= gamma

    cone.scaling_updated = true
end

function update_hess(cone::EpiNormEucl)
    @assert cone.grad_updated
    @assert cone.is_feas
    if cone.use_scaling
        if !cone.scaling_updated
            update_scaling(cone)
        end
        w = cone.w

        # dumb instantiation
        Wbar = similar(cone.inv_hess)
        Wbar.data[1, 1] = w[1]
        Wbar.data[1, 2:end] .= w[2:end]
        Wbar.data[2:end, 2:end] = w[2:end] * w[2:end]' / (w[1] + 1)
        Wbar.data[2:end, 2:end] += I
        cone.inv_hess.data .= Wbar * Wbar / (cone.dist / cone.dual_dist) ^ (1 / 2)

        # TODO faster way
    else
        mul!(cone.hess.data, 2 * cone.grad, cone.grad')
        cone.hess += inv(cone.dist) * I
        cone.hess[1, 1] -= 2 / cone.dist
        cone.hess_updated = true
    end
    return cone.hess
end

function update_inv_hess(cone::EpiNormEucl)
    @assert cone.is_feas
    if cone.use_scaling
        if !cone.scaling_updated
            update_scaling(cone)
        end
        w = cone.w

        # dumb instantiation
        Wbar = similar(cone.inv_hess)
        Wbar.data[1, :] .= w
        Wbar.data[2:end, 2:end] = w[2:end] * w[2:end]' / (w[1] + 1)
        Wbar.data[2:end, 2:end] += I
        cone.inv_hess.data .= Wbar * Wbar * (cone.dist / cone.dual_dist) ^ (1 / 2)

        # more efficiently (from ECOS)

        # NOTE only forms upper triangle
        # cone.inv_hess[1, 1] = w[1]
        # ww = sum(abs2, w[2:end])
        #
        # rel_w_dist = ww / abs2(1 + w[1])
        # # NOTE not sure about this constant
        # b = 1 + 2 / (1 + w[1]) + rel_w_dist
        # cone.inv_hess[1, 2:end] = b * w[2:end]
        # c = 1 + w[1] + rel_w_dist
        #
        # cone.inv_hess.data[2:end, 2:end] = w[2:end] * w[2:end]' * c
        # cone.inv_hess.data[2:end, 2:end] += I
        # cone.inv_hess.data .*= sqrt(dist / dual_dist)
    else
        mul!(cone.inv_hess.data, 2 * cone.point, cone.point')
        cone.inv_hess += cone.dist * I
        cone.inv_hess[1, 1] -= 2 * cone.dist
        cone.inv_hess_updated = true
    end
    return cone.inv_hess
end

update_hess_prod(cone::EpiNormEucl) = nothing
update_inv_hess_prod(cone::EpiNormEucl) = nothing

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas
    if cone.use_scaling
        # TODO
        hess = update_hess(cone)
        prod = hess * arr
    else
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
    end
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas
    if cone.use_scaling
        # TODO
        inv_hess = update_inv_hess(cone)
        prod = inv_hess * arr
    else
        @inbounds for j in 1:size(prod, 2)
            @views pa = dot(cone.point, arr[:, j])
            @. prod[:, j] = pa * cone.point * 2
        end
        @. @views prod[1, :] -= cone.dist * arr[1, :]
        @. @views prod[2:end, :] += cone.dist * arr[2:end, :]
    end
    return prod
end

# multiplies arr by W, the squareroot of the scaling matrix
# TODO there is a faster way
function scalmat_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    if !cone.scaling_updated
        update_scaling(cone)
    end
    w = cone.w
    dist = cone.dist
    dual_dist = cone.dual_dist

    W = w * w'
    W[2:end, 2:end] ./= (w[1] + 1)
    W[2:end, 2:end] += I
    W .*= (dist / dual_dist) ^ (1 / 4)

    prod .= W * arr
    return prod
end

# divides arr by lambda, the scaled point
# TODO there is a faster way
function scalvec_ldiv!(div, cone::EpiNormEucl, arr)
    if !cone.scaling_updated
        update_scaling(cone)
    end

    gamma = cone.gamma
    scaled_point = cone.scaled_point
    scaled_dual_point = cone.scaled_dual_point
    dist = cone.dist
    dual_dist = cone.dual_dist

    lambda_scaled = similar(cone.point)

    lambda_scaled[1] = gamma
    lambda_scaled[2:end] = (gamma + scaled_dual_point[1]) * scaled_point[2:end] + (gamma + scaled_point[1]) * scaled_dual_point[2:end]
    lambda_scaled[2:end] /= (scaled_point[1] + scaled_dual_point[1] + 2 * gamma)

    lambda = sqrt(sqrt(dist * dual_dist)) * lambda_scaled

    return jordan_ldiv(div, lambda, arr)

end

function jordan_ldiv(C::AbstractVecOrMat, A::Vector, B::AbstractVecOrMat)
    m = length(A)
    @assert m == size(B, 1)
    @assert size(B) == size(C)
    A1 = A[1]
    A2m = view(A, 2:m)
    schur = abs2(A1) - sum(abs2, A2m)
    @views begin
        mul!(C[1, :], B[2:end, :]', A2m, true, true)
        @. C[2:end, :] = A2m * C[1, :]' / A1
        axpby!(A1, B[1, :], -1.0, C[1, :])
        @. C[2:end, :] -= A2m * B[1, :]'
        C ./= schur
        @. C[2:end, :] += B[2:end, :] / A1
    end
    return C
end
