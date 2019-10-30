#=
Copyright 2018, Chris Coey and contributors

TODO describe hermitian complex PSD cone
on-diagonal (real) elements have one slot in the vector and below diagonal (complex) elements have two consecutive slots in the vector

row-wise lower triangle of positive semidefinite matrix cone
W \in S^n : 0 >= eigmin(W)
(see equivalent MathOptInterface PositiveSemidefiniteConeTriangle definition)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-logdet(W)

TODO fix native and moi tests, and moi
=#

mutable struct PosSemidefTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_scaling::Bool
    dim::Int
    side::Int
    is_complex::Bool
    point::Vector{T}
    dual_point::Vector{T}
    rt2::T

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    scaling_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    mat::Matrix{R}
    dual_mat::Matrix{R}
    mat2::Matrix{R}
    dual_mat2::Matrix{R}
    mat3::Matrix{R}
    mat4::Matrix{R}
    inv_mat::Matrix{R}
    fact_mat
    dual_fact_mat

    scalmat_sqrt::Matrix{R}
    scalmat_sqrti::Matrix{R}
    lambda::Vector{R} # TODO remove if unneeded

    function PosSemidefTri{T, R}(dim::Int, use_scaling::Bool = true) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert dim >= 1
        cone = new{T, R}()
        cone.dim = dim # real vector dimension
        cone.rt2 = sqrt(T(2))
        cone.use_scaling = use_scaling
        if R <: Complex
            side = isqrt(dim) # real lower triangle and imaginary under diagonal
            @assert side^2 == dim
            cone.is_complex = true
        else
            side = round(Int, sqrt(0.25 + 2 * dim) - 0.5) # real lower triangle
            @assert side * (side + 1) == 2 * dim
            cone.is_complex = false
        end
        cone.side = side
        return cone
    end
end

use_dual(cone::PosSemidefTri) = false # self-dual

use_scaling(cone::PosSemidefTri) = cone.use_scaling

load_dual_point(cone::PosSemidefTri, dual_point::AbstractVector) = copyto!(cone.dual_point, dual_point)

reset_data(cone::PosSemidefTri) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.scaling_updated = false)

function setup_data(cone::PosSemidefTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.lambda = zeros(T, cone.side)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.mat = zeros(R, cone.side, cone.side)
    cone.dual_mat = similar(cone.mat)
    cone.mat2 = similar(cone.mat)
    cone.dual_mat2 = similar(cone.mat)
    cone.mat3 = similar(cone.mat)
    cone.mat4 = similar(cone.mat)
    return
end

get_nu(cone::PosSemidefTri) = cone.side

function set_initial_point(arr::AbstractVector, cone::PosSemidefTri)
    incr = (cone.is_complex ? 2 : 1)
    arr .= 0
    k = 1
    @inbounds for i in 1:cone.side
        arr[k] = 1
        k += incr * i + 1
    end
    return arr
end

function update_feas(cone::PosSemidefTri)
    @assert !cone.feas_updated
    svec_to_smat!(cone.mat, cone.point, cone.rt2)
    copyto!(cone.mat2, cone.mat)
    cone.fact_mat = cholesky!(Hermitian(cone.mat2, :U), check = false)
    cone.is_feas = isposdef(cone.fact_mat)
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::PosSemidefTri)
    @assert cone.is_feas
    cone.inv_mat = inv(cone.fact_mat)
    smat_to_svec!(cone.grad, cone.inv_mat, cone.rt2)
    cone.grad .*= -1
    copytri!(cone.mat, 'U', cone.is_complex)
    cone.grad_updated = true
    return cone.grad
end

function update_scaling(cone::PosSemidefTri)
    @assert !cone.scaling_updated
    @assert cone.is_feas
    fact = cone.fact_mat
    svec_to_smat!(cone.dual_mat, cone.dual_point, cone.rt2)
    copyto!(cone.dual_mat2, cone.dual_mat)
    dual_fact = cone.dual_fact_mat = cholesky!(Hermitian(cone.dual_mat2, :U), check = false)
    @assert isposdef(cone.dual_fact_mat)

    (U, lambda, V) = svd(dual_fact.U * fact.L)
    cone.scalmat_sqrt = fact.L * V * Diagonal(sqrt.(inv.(lambda)))
    cone.scalmat_sqrti = Diagonal(inv.(sqrt.(lambda))) * U' * dual_fact.U
    cone.lambda = lambda

    cone.scaling_updated = true
    return cone.scaling_updated
end

# TODO parallelize
function _build_hess(H::Matrix{T}, mat::Matrix{T}, rt2::T) where {T <: Real}
    side = size(mat, 1)
    k = 1
    for i in 1:side, j in 1:i
        k2 = 1
        @inbounds for i2 in 1:side, j2 in 1:i2
            if (i == j) && (i2 == j2)
                H[k2, k] = abs2(mat[i2, i])
            elseif (i != j) && (i2 != j2)
                H[k2, k] = mat[i2, i] * mat[j, j2] + mat[j2, i] * mat[j, i2]
            else
                H[k2, k] = rt2 * mat[i2, i] * mat[j, j2]
            end
            if k2 == k
                break
            end
            k2 += 1
        end
        k += 1
    end
    return H
end

function _build_hess(H::Matrix{T}, mat::Matrix{Complex{T}}, rt2::T) where {T <: Real}
    side = size(mat, 1)
    k = 1
    for i in 1:side, j in 1:i
        k2 = 1
        if i == j
            @inbounds for i2 in 1:side, j2 in 1:i2
                if i2 == j2
                    H[k2, k] = abs2(mat[i2, i])
                    k2 += 1
                else
                    c = rt2 * mat[i, i2] * mat[j2, j]
                    H[k2, k] = real(c)
                    k2 += 1
                    H[k2, k] = -imag(c)
                    k2 += 1
                end
                if k2 > k
                    break
                end
            end
            k += 1
        else
            @inbounds for i2 in 1:side, j2 in 1:i2
                if i2 == j2
                    c = rt2 * mat[i2, i] * mat[j, j2]
                    H[k2, k] = real(c)
                    H[k2, k + 1] = -imag(c)
                    k2 += 1
                else
                    b1 = mat[i2, i] * mat[j, j2]
                    b2 = mat[j2, i] * mat[j, i2]
                    c1 = b1 + b2
                    H[k2, k] = real(c1)
                    H[k2, k + 1] = -imag(c1)
                    k2 += 1
                    c2 = b1 - b2
                    H[k2, k] = imag(c2)
                    H[k2, k + 1] = real(c2)
                    k2 += 1
                end
                if k2 > k
                    break
                end
            end
            k += 2
        end
    end
    return H
end

function update_hess(cone::PosSemidefTri)
    @assert cone.grad_updated
    if cone.use_scaling
        if !cone.scaling_updated
            update_scaling(cone)
        end
        _build_hess(cone.hess.data, cone.scalmat_sqrti' * cone.scalmat_sqrti, cone.rt2)
    else
        _build_hess(cone.hess.data, cone.inv_mat, cone.rt2)
    end
    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::PosSemidefTri)
    @assert is_feas(cone)
    if cone.use_scaling
        if !cone.scaling_updated
            update_scaling(cone)
        end
        _build_hess(cone.inv_hess.data, cone.scalmat_sqrt * cone.scalmat_sqrt', cone.rt2)
    else
        _build_hess(cone.inv_hess.data, cone.mat, cone.rt2)
    end
    cone.inv_hess_updated = true
    return cone.inv_hess
end

update_hess_prod(cone::PosSemidefTri) = nothing
update_inv_hess_prod(cone::PosSemidefTri) = nothing

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    @assert is_feas(cone)
    if cone.use_scaling
        if !cone.scaling_updated
            update_scaling(cone)
        end
        @inbounds for i in 1:size(arr, 2)
            svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
            mul!(cone.mat3, Hermitian(cone.mat4, :U), cone.scalmat_sqrti' * cone.scalmat_sqrti)
            mul!(cone.mat4, Hermitian(cone.scalmat_sqrti' * cone.scalmat_sqrti, :U), cone.mat3)
            smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
        end
    else
        @inbounds for i in 1:size(arr, 2)
            svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
            copytri!(cone.mat4, 'U', cone.is_complex)
            rdiv!(cone.mat4, cone.fact_mat)
            ldiv!(cone.fact_mat, cone.mat4)
            smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
        end
    end
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    @assert is_feas(cone)
    if cone.use_scaling
        if !cone.scaling_updated
            update_scaling(cone)
        end
        @inbounds for i in 1:size(arr, 2)
            svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
            mul!(cone.mat3, Hermitian(cone.mat4, :U), cone.scalmat_sqrt * cone.scalmat_sqrt')
            mul!(cone.mat4, Hermitian(cone.scalmat_sqrt * cone.scalmat_sqrt', :U), cone.mat3)
            smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
        end
    else
        @inbounds for i in 1:size(arr, 2)
            svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
            mul!(cone.mat3, Hermitian(cone.mat4, :U), cone.mat)
            mul!(cone.mat4, Hermitian(cone.mat, :U), cone.mat3)
            smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
        end
    end
    return prod
end

# TODO since outer-producting with non-symmetric matrices, would make sense to factorize arr and syrk
function scalmat_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    if !cone.scaling_updated
        update_scaling(cone)
    end
    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
        mul!(cone.mat3, Hermitian(cone.mat4, :U), cone.scalmat_sqrt)
        mul!(cone.mat4, cone.scalmat_sqrt', cone.mat3)
        smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
    end
    return prod
end

function scalmat_ldiv!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    if !cone.scaling_updated
        update_scaling(cone)
    end
    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
        mul!(cone.mat3, Hermitian(cone.mat4, :U), cone.scalmat_sqrti')
        mul!(cone.mat4, cone.scalmat_sqrti, cone.mat3)
        smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
    end
    return prod
end

function scalvec_ldiv!(div::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    if !cone.scaling_updated
        update_scaling(cone)
    end
    @. cone.mat2 = cone.lambda
    @. cone.mat2 += cone.mat2'
    @. cone.mat2 = 2 / cone.mat2
    svec_to_smat!(cone.mat3, arr, cone.rt2)
    @. cone.mat4 = cone.mat3 * cone.mat2
    smat_to_svec!(div, cone.mat4, cone.rt2)
    # @show sqrt(cone.point[1] * cone.dual_point[1])
    return div
end

function conic_prod!(w::AbstractVector, u::AbstractVector, v::AbstractVector, cone::PosSemidefTri)
    U = Hermitian(svec_to_smat!(cone.mat2, u, cone.rt2), :U)
    V = Hermitian(svec_to_smat!(cone.mat3, v, cone.rt2), :U)
    W = cone.mat4
    W .= (U * V + V' * U') / 2
    smat_to_svec!(w, W, cone.rt2)
    return w
end

# dist = one(T)
# @inbounds for i in eachindex(point)
#     if dir[i] < 0
#         dist = min(dist, -point[i] / dir[i])
#     end
# end
# return dist

function dist_to_bndry(cone::PosSemidefTri{T, R}, fact, dir::AbstractVector{T}) where {R <: RealOrComplex{T}} where {T <: Real}
    dist = one(T)
    dir_mat = Hermitian(svec_to_smat!(cone.mat2, dir, cone.rt2), :U)
    eig_vals = eigvals(inv(fact.L) * dir_mat * inv(fact.L)')
    @inbounds for v in eig_vals
        if v < 0
            dist = min(dist, -inv(v))
        end
    end
    return dist
end

function step_max_dist(cone::PosSemidefTri, s_sol::AbstractVector, z_sol::AbstractVector)
    # TODO only need this for dual_fact_mat, here and in other cones cones maybe break up update_scaling
    if !cone.scaling_updated
        update_scaling(cone)
    end
    # TODO this could go in Cones.jl
    # Stilde = scalmat_ldiv!(cone.mat2, cone.point, cone)
    # Ztilde = scalmat_prod!(cone.mat2, cone.dual_point, cone)

    primal_dist = dist_to_bndry(cone, cone.fact_mat, s_sol)
    dual_dist = dist_to_bndry(cone, cone.dual_fact_mat, z_sol)
    step_dist = min(primal_dist, dual_dist)
end

# TODO fix later, rt2::T doesn't work with tests using ForwardDiff
function smat_to_svec!(vec::AbstractVector{T}, mat::AbstractMatrix{T}, rt2::Number) where {T}
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            vec[k] = mat[i, j]
        else
            vec[k] = mat[i, j] * rt2
        end
        k += 1
    end
    return vec
end

function svec_to_smat!(mat::AbstractMatrix{T}, vec::AbstractVector{T}, rt2::Number) where {T}
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            mat[i, j] = vec[k]
        else
            mat[i, j] = vec[k] / rt2
        end
        k += 1
    end
    return mat
end

function smat_to_svec!(vec::AbstractVector{T}, mat::AbstractMatrix{Complex{T}}, rt2::Number) where {T}
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            vec[k] = real(mat[i, j])
            k += 1
        else
            ck = mat[i, j] * rt2
            vec[k] = real(ck)
            k += 1
            vec[k] = -imag(ck)
            k += 1
        end
    end
    return vec
end

function svec_to_smat!(mat::AbstractMatrix{Complex{T}}, vec::AbstractVector{T}, rt2::Number) where {T}
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            mat[i, j] = vec[k]
            k += 1
        else
            mat[i, j] = Complex(vec[k], -vec[k + 1]) / rt2
            k += 2
        end
    end
    return mat
end
