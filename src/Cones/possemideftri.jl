#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
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
    fact_mat::Matrix{R}
    dual_fact_mat::Matrix{R}
    work_mat::Matrix{R}
    work_mat2::Matrix{R}
    work_mat3::Matrix{R}
    inv_mat::Matrix{R}
    fact
    dual_fact

    scalmat_sqrt::Matrix{R}
    scalmat_sqrti::Matrix{R}
    lambda::Vector{R}
    bndry_dists::Vector{R}
    correction::Vector{R}

    function PosSemidefTri{T, R}(dim::Int; use_scaling::Bool = true) where {R <: RealOrComplex{T}} where {T <: Real}
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
    cone.bndry_dists = zeros(T, cone.side)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.mat = zeros(R, cone.side, cone.side)
    cone.dual_mat = similar(cone.mat)
    cone.fact_mat = similar(cone.mat)
    cone.dual_fact_mat = similar(cone.mat)
    cone.work_mat = similar(cone.mat)
    cone.work_mat2 = similar(cone.mat)
    cone.work_mat3 = similar(cone.mat)
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
    copyto!(cone.fact_mat, cone.mat)
    cone.fact = cholesky!(Hermitian(cone.fact_mat, :U), check = false)
    cone.is_feas = isposdef(cone.fact)
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::PosSemidefTri)
    @assert cone.is_feas
    cone.inv_mat = inv(cone.fact)
    smat_to_svec!(cone.grad, cone.inv_mat, cone.rt2)
    cone.grad .*= -1
    copytri!(cone.mat, 'U', cone.is_complex)
    cone.grad_updated = true
    return cone.grad
end

function update_scaling(cone::PosSemidefTri)
    @assert !cone.scaling_updated
    @assert cone.is_feas
    fact = cone.fact
    svec_to_smat!(cone.dual_mat, cone.dual_point, cone.rt2)
    copyto!(cone.dual_fact_mat, cone.dual_mat)
    dual_fact = cone.dual_fact = cholesky!(Hermitian(cone.dual_fact_mat, :U), check = false)
    @assert isposdef(cone.dual_fact)

    # TODO preallocate
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

# PXP where P is Hermitian
function herm_congruence_prod!(prod::AbstractVecOrMat, inner::AbstractVecOrMat, outer::AbstractVecOrMat, cone::PosSemidefTri)
    @inbounds for i in 1:size(inner, 2)
        svec_to_smat!(cone.work_mat2, view(inner, :, i), cone.rt2)
        mul!(cone.work_mat, Hermitian(cone.work_mat2, :U), outer)
        mul!(cone.work_mat2, Hermitian(outer, :U), cone.work_mat)
        smat_to_svec!(view(prod, :, i), cone.work_mat2, cone.rt2)
    end
    return prod
end

# PXP or P'XP depending on `trans`, where P is not symmetric or Hermitian
function gen_congruence_prod!(prod::AbstractVecOrMat, inner::AbstractVecOrMat, outer::AbstractVecOrMat, cone::PosSemidefTri)
    @inbounds for i in 1:size(inner, 2)
        svec_to_smat!(cone.work_mat2, view(inner, :, i), cone.rt2)
        mul!(cone.work_mat, Hermitian(cone.work_mat2, :U), outer)
        mul!(cone.work_mat2, outer', cone.work_mat)
        smat_to_svec!(view(prod, :, i), cone.work_mat2, cone.rt2)
    end
    return prod
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    @assert is_feas(cone)
    if cone.use_scaling
        if !cone.scaling_updated
            update_scaling(cone)
        end
        mul!(cone.work_mat3, cone.scalmat_sqrti', cone.scalmat_sqrti)
        herm_congruence_prod!(prod, arr, cone.work_mat3, cone)
    else
        @inbounds for i in 1:size(arr, 2)
            svec_to_smat!(cone.work_mat2, view(arr, :, i), cone.rt2)
            copytri!(cone.work_mat2, 'U', cone.is_complex)
            rdiv!(cone.work_mat2, cone.fact)
            ldiv!(cone.fact, cone.work_mat2)
            smat_to_svec!(view(prod, :, i), cone.work_mat2, cone.rt2)
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
        mul!(cone.work_mat3, cone.scalmat_sqrt, cone.scalmat_sqrt')
        herm_congruence_prod!(prod, arr, cone.work_mat3, cone)
    else
        herm_congruence_prod!(prod, arr, cone.mat, cone)
    end
    return prod
end

# TODO think about whether transpose oracle is needed
function scalmat_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    if !cone.scaling_updated
        update_scaling(cone)
    end
    gen_congruence_prod!(prod, arr, cone.scalmat_sqrt, cone)
    return prod
end

# TODO think about whether transpose oracle is needed in the future (it is now)
function scalmat_ldiv!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri; trans::Bool = false)
    if !cone.scaling_updated
        update_scaling(cone)
    end
    outer_term = (trans ? cone.scalmat_sqrti' : cone.scalmat_sqrti)
    gen_congruence_prod!(prod, arr, outer_term, cone)
    return prod
end

function scalvec_ldiv!(div::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    if !cone.scaling_updated
        update_scaling(cone)
    end
    @. cone.work_mat = cone.lambda
    @. cone.work_mat += cone.work_mat'
    @. cone.work_mat = 2 / cone.work_mat
    svec_to_smat!(cone.work_mat2, arr, cone.rt2)
    # only upper triangle of cone.work_mat is updated, but that is enough (wrapping in UpperTriangular is slower)
    @. cone.work_mat3 = cone.work_mat * cone.work_mat2
    smat_to_svec!(div, cone.work_mat3, cone.rt2)
    return div
end

function conic_prod!(w::AbstractVector, u::AbstractVector, v::AbstractVector, cone::PosSemidefTri)
    U = Hermitian(svec_to_smat!(cone.work_mat, u, cone.rt2), :U)
    V = Hermitian(svec_to_smat!(cone.work_mat2, v, cone.rt2), :U)
    W = cone.work_mat3
    mul!(W, U, V)
    @. W = (W + W') / 2
    smat_to_svec!(w, W, cone.rt2)
    return w
end

function dist_to_bndry(cone::PosSemidefTri{T, R}, fact, dir::AbstractVector{T}) where {R <: RealOrComplex{T}} where {T <: Real}
    svec_to_smat!(cone.work_mat2, dir, cone.rt2)
    mul!(cone.work_mat, Hermitian(cone.work_mat2, :U), inv(fact.U))
    ldiv!(fact.U', cone.work_mat)
    # TODO preallocate. also explore faster options.
    # NOTE julia calls eigvals inside eigmin, and eigmin is not currently implemented in GenericLinearAlgebra
    v = eigvals(Hermitian(cone.work_mat, :U))
    inv_min_dist = minimum(v)
    if inv_min_dist >= 0
        return T(Inf)
    else
        return -inv(inv_min_dist)
    end
end

function step_max_dist(cone::PosSemidefTri, s_sol::AbstractVector, z_sol::AbstractVector)
    # TODO only need this for dual_fact, here and in other cones cones maybe break up update_scaling
    @assert cone.is_feas
    if !cone.scaling_updated
        update_scaling(cone)
    end
    # TODO this could go in Cones.jl
    primal_dist = dist_to_bndry(cone, cone.fact, s_sol)
    dual_dist = dist_to_bndry(cone, cone.dual_fact, z_sol)
    step_dist = min(primal_dist, dual_dist)
    return step_dist
end

# TODO refactor into Cones.jl
function correction(cone::PosSemidefTri, s_sol::AbstractVector, z_sol::AbstractVector)
    @assert cone.grad_updated
    tmp_s = scalmat_ldiv!(similar(s_sol), s_sol, cone)
    tmp_z = scalmat_prod!(similar(z_sol), z_sol, cone)
    mehrotra_term = conic_prod!(similar(cone.point), tmp_s, tmp_z, cone)

    C = scalvec_ldiv!(similar(cone.point), mehrotra_term, cone)
    scalmat_ldiv!(cone.correction, C, cone)

    return cone.correction
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
