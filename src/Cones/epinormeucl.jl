#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of Euclidean (2-)norm (AKA second-order cone)
(u in R, w in R^n) : u >= norm_2(w)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd, halved
-log(u^2 - norm_2(w)^2) / 2

contains some adapted code from https://github.com/embotech/ecos
TODO tidy up modifications due to halving barrier
TODO all oracles assumed we are using dense version of the Nesterov Todd scaling
TODO factor out repetitions in products, hessians, inverse hessians
TODO factor out eta
=#

mutable struct EpiNormEucl{T <: Real} <: Cone{T}
    use_scaling::Bool
    dim::Int
    point::Vector{T}
    dual_point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    scaling_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    scaled_point::Vector{T}
    scaled_dual_point::Vector{T}
    w::Vector{T} #think about naming, it's w_bar in cvxopt paper
    dist::T
    dual_dist::T
    gamma::T
    ww::T
    b::T
    c::T
    correction::Vector{T}

    function EpiNormEucl{T}(dim::Int; use_scaling::Bool = true) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.dim = dim
        cone.use_scaling = use_scaling
        return cone
    end
end

use_dual(cone::EpiNormEucl) = false # self-dual

use_scaling(cone::EpiNormEucl) = cone.use_scaling

load_dual_point(cone::EpiNormEucl, dual_point::AbstractVector) = copyto!(cone.dual_point, dual_point)

reset_data(cone::EpiNormEucl) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.scaling_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::EpiNormEucl{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.scaled_point = zeros(T, dim)
    cone.scaled_dual_point = zeros(T, dim)
    cone.w = zeros(T, dim)
    cone.correction = zeros(T, dim)
    return
end

get_nu(cone::EpiNormEucl) = 1

function set_initial_point(arr::AbstractVector, cone::EpiNormEucl)
    arr .= 0
    arr[1] = 1
    return arr
end

function check_feas(cone::EpiNormEucl, point::Vector, primal::Bool)
    u = point[1]
    if u > 0
        w = view(point, 2:cone.dim)
        dist = abs2(u) - sum(abs2, w)
        # TODO record dual_dist here and make sure operations trickle through correctly
        primal ? (cone.dist = dist) : nothing
        return dist > 0
    else
        return false
    end
end

function update_feas(cone::EpiNormEucl)
    @assert !cone.feas_updated
    cone.is_feas = check_feas(cone, cone.point, true)
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
    cone.gamma = sqrt((1 + dot(scaled_point, scaled_dual_point)) / 2)

    w = cone.w
    w[1] = scaled_point[1] + scaled_dual_point[1]
    @. @views w[2:end] = scaled_point[2:end] - scaled_dual_point[2:end]
    w ./= 2 # NOTE probably not /2 for unhalved barrier
    w ./= cone.gamma

    w2nw2n = sum(abs2, w[2:end])
    cone.ww = abs2(w[1]) + w2nw2n
    cone.b = 1 + 2 / (1 + w[1]) + w2nw2n / abs2(1 + w[1])
    cone.c = 1 + w[1] + w2nw2n / (1 + w[1])

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
        hess = cone.hess.data

        # dumb instantiation
        # Wbar = similar(cone.hess)
        # Wbar.data[1, 1] = w[1]
        # Wbar.data[1, 2:end] .= -w[2:end]
        # Wbar.data[2:end, 2:end] = w[2:end] * w[2:end]' / (w[1] + 1)
        # Wbar.data[2:end, 2:end] += I
        # cone.hess.data .= Wbar * Wbar * (cone.dual_dist / cone.dist) ^ (1 / 2)

        # pattern matched from inverse Hessian
        hess[1, 1] = cone.ww
        @. hess[1, 2:end] = -cone.c * w[2:end]
        @views mul!(hess[2:end, 2:end], w[2:end], w[2:end]', cone.b, false)
        hess[2:end, 2:end] += I
        hess .*= sqrt(cone.dual_dist / cone.dist)
    else
        mul!(cone.hess.data, cone.grad, cone.grad', 2, false)
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
        inv_hess = cone.inv_hess.data

        # dumb instantiation
        # Wbar = similar(cone.inv_hess)
        # Wbar.data[1, :] .= w
        # Wbar.data[2:end, 2:end] = w[2:end] * w[2:end]' / (w[1] + 1)
        # Wbar.data[2:end, 2:end] += I
        # cone.inv_hess.data .= Wbar * Wbar * (cone.dist / cone.dual_dist) ^ (1 / 2)

        # see https://github.com/embotech/ecos/blob/develop/src/kkt.c
        inv_hess[1, 1] = cone.ww
        @. inv_hess[1, 2:end] = cone.c * w[2:end]
        @views mul!(inv_hess[2:end, 2:end], w[2:end], w[2:end]', cone.b, false)
        inv_hess[2:end, 2:end] += I
        inv_hess .*= sqrt(cone.dist / cone.dual_dist)
    else
        mul!(cone.inv_hess.data, cone.point, cone.point', 2, false)
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
        if !cone.scaling_updated
            update_scaling(cone)
        end
        w = cone.w
        prod[1, :] = cone.ww * arr[1, :] - cone.c * arr[2:end, :]' * w[2:end]
        for j in 1:size(prod, 2)
            @views pa = dot(cone.w[2:end], arr[2:end, j])
            @. prod[2:end, j] = pa * cone.w[2:end] * cone.b
        end
        @. prod[2:end, :] += arr[2:end, :]
        @. prod[2:end, :] -= cone.c * arr[1, :]' * w[2:end]
        prod .*= sqrt(cone.dual_dist / cone.dist)

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
        if !cone.scaling_updated
            update_scaling(cone)
        end
        w = cone.w
        prod[1, :] = cone.ww * arr[1, :] + cone.c * arr[2:end, :]' * w[2:end]
        for j in 1:size(prod, 2)
            @views pa = dot(cone.w[2:end], arr[2:end, j])
            @. prod[2:end, j] = pa * cone.w[2:end] * cone.b
        end
        @. prod[2:end, :] += arr[2:end, :]
        @. prod[2:end, :] += cone.c * arr[1, :]' * w[2:end]
        prod .*= sqrt(cone.dist / cone.dual_dist)
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

    prod[1, :] .= arr' * w
    @views begin
        for j in 1:size(prod, 2)
            pa = dot(w[2:end], arr[2:end, j])
            @. prod[2:end, j] = w[2:end] * pa
        end
        @. prod[2:end, :] /= (w[1] + 1)
        @. prod[2:end, :] += arr[2:end, :]
        @. prod[2:end, :] += arr[1, :]' * w[2:end]
    end
    prod .*= (cone.dist / cone.dual_dist) ^ (1 / 4)

    # Wbar = similar(cone.inv_hess)
    # Wbar.data[1, :] .= w
    # Wbar.data[2:end, 2:end] = w[2:end] * w[2:end]' / (w[1] + 1)
    # Wbar.data[2:end, 2:end] += I
    # W = Wbar * (cone.dist / cone.dual_dist) ^ (1 / 4)
    # prod .= W * arr

    return prod
end

function scalmat_ldiv!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    if !cone.scaling_updated
        update_scaling(cone)
    end
    w = cone.w

    # prod[1, :] .= arr' * w
    @views begin
        prod[1, :] .= arr[1, :] * w[1] .- arr[2:end, :]' * w[2:end]
        for j in 1:size(prod, 2)
            pa = dot(w[2:end], arr[2:end, j])
            @. prod[2:end, j] = w[2:end] * pa
        end
        @. prod[2:end, :] /= (w[1] + 1)
        @. prod[2:end, :] += arr[2:end, :]
        @. prod[2:end, :] -= arr[1, :]' * w[2:end]
    end
    prod .*= (cone.dual_dist / cone.dist) ^ (1 / 4)
    return prod
end

# returns  W_inv \circ lambda_inv \circ correction = -grad \circ correction
function correction(cone::EpiNormEucl, s_sol::AbstractVector, z_sol::AbstractVector)
    tmp_s = similar(s_sol)
    tmp_z = similar(z_sol)
    tmp = similar(tmp_s)
    scalmat_ldiv!(tmp_s, s_sol, cone)
    scalmat_prod!(tmp_z, z_sol, cone)

    tmp[1] = dot(tmp_s, tmp_z)
    @. @views tmp[2:end] = tmp_s[1] * tmp_z[2:end] + tmp_z[1] * tmp_s[2:end]

    cone.correction[1] = -dot(cone.grad, tmp)
    @. @views cone.correction[2:end] = -cone.grad[1] * tmp[2:end] - tmp[1] * cone.grad[2:end]

    return cone.correction
end

# divides arr by lambda, the scaled point
# TODO there is a faster way
# TODO remove this oracle if not used in the near future
function scalvec_ldiv!(div::AbstractVecOrMat, cone::EpiNormEucl, arr::AbstractVecOrMat)
    if !cone.scaling_updated
        update_scaling(cone)
    end

    gamma = cone.gamma
    scaled_point = cone.scaled_point
    scaled_dual_point = cone.scaled_dual_point
    dist = cone.dist
    dual_dist = cone.dual_dist

    lambda_scaled = similar(cone.point)

    # lambda_scaled[1] = gamma
    # lambda_scaled[2:end] = (gamma + scaled_dual_point[1]) * scaled_point[2:end] + (gamma + scaled_point[1]) * scaled_dual_point[2:end]
    # lambda_scaled[2:end] /= (scaled_point[1] + scaled_dual_point[1] + 2 * gamma)
    scalmat_prod!(lambda_scaled, cone.scaled_dual_point, cone)

    lambda = (dist * dual_dist) ^ (1 / 4) * lambda_scaled
    # @show lambda
    #
    # @show jordan_ldiv!(div, lambda, arr)
    # @show arr
    # @show Symmetric([lambda[1] lambda[2]; lambda[2] lambda[1]]) \ arr

    return jordan_ldiv!(div, lambda, arr)

end

function jordan_ldiv!(C::AbstractVecOrMat, A::Vector, B::AbstractVecOrMat)
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

function dist_to_bndry(::EpiNormEucl{T}, lambda::Vector{T}, dir::AbstractVector{T}) where {T}
    lambda_dir_dist = lambda[1] * dir[1] - sum(lambda[i] * dir[i] for i in 2:length(lambda))
    fact = (lambda_dir_dist + dir[1]) / (lambda[1] + 1)

    dist2n = zero(T)
    for i in 2:length(lambda)
        dist2n += abs2(dir[i] - fact * lambda[i])
    end
    return -lambda_dir_dist + sqrt(dist2n)
end

function step_max_dist(cone::EpiNormEucl{T}, s_sol::AbstractVector{T}, z_sol::AbstractVector{T}) where {T}
    lambda = similar(cone.point)
    s_sol_scaled = similar(cone.point)
    z_sol_scaled = similar(cone.point)

    point = cone.point
    dual_point = cone.dual_point

    # get lambda
    # TODO there is a shortcut for getting lambda
    scalmat_prod!(lambda, cone.dual_point, cone)
    scalmat_ldiv!(s_sol_scaled, s_sol, cone)
    scalmat_prod!(z_sol_scaled, z_sol, cone)

    lambda_dist = abs2(lambda[1]) - sum(abs2, lambda[2:end])
    lambda_dist_sqrt = sqrt(lambda_dist)
    lambda ./= lambda_dist_sqrt
    primal_step_dist = lambda_dist_sqrt / dist_to_bndry(cone, lambda, s_sol_scaled)
    dual_step_dist = lambda_dist_sqrt / dist_to_bndry(cone, lambda, z_sol_scaled)

    # TODO refactor
    step_dist = one(T)
    if primal_step_dist > 0
        step_dist = min(step_dist, primal_step_dist)
    end
    if dual_step_dist > 0
        step_dist = min(step_dist, dual_step_dist)
    end

    return step_dist

end
