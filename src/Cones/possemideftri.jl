#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

row-wise lower triangle of positive semidefinite matrix cone (scaled "svec" form)
W \in S^n : 0 >= eigmin(W)
NOTE on-diagonal (real) elements have one slot in the vector and below diagonal (complex) elements have two consecutive slots in the vector

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-logdet(W)

TODO
- describe svec scaling
- describe hermitian complex PSD cone
- fix native and moi tests, and moi wrapper
=#

mutable struct PosSemidefTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_scaling::Bool
    use_3order_corr::Bool
    try_scaled_updates::Bool # run algorithm in scaled variables for numerical reasons TODO decide whether to keep this as an option
    dim::Int
    side::Int
    is_complex::Bool
    point::Vector{T}
    dual_point::Vector{T}

    prev_scal_point::Vector{T}
    prev_scal_dual_point::Vector{T}
    new_scal_point::Vector{T} # NOTE v in MOSEK; always diagonal, but stored as a vector for stepping
    rt2::T

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
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

    # factorizations of the scaling matrix
    scalmat_sqrt::Matrix{R}
    scalmat_sqrti::Matrix{R}
    correction::Vector{T}

    function PosSemidefTri{T, R}(
        dim::Int;
        use_scaling::Bool = true,
        use_3order_corr::Bool = true,
        try_scaled_updates::Bool = false,
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert dim >= 1
        cone = new{T, R}()
        cone.dim = dim # real vector dimension
        cone.rt2 = sqrt(T(2))
        cone.use_scaling = use_scaling
        cone.use_3order_corr = use_3order_corr
        cone.try_scaled_updates = try_scaled_updates
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

use_3order_corr(cone::PosSemidefTri) = cone.use_3order_corr

try_scaled_updates(cone::PosSemidefTri) = cone.try_scaled_updates # TODO

load_dual_point(cone::PosSemidefTri, dual_point::AbstractVector) = copyto!(cone.dual_point, dual_point)

reset_data(cone::PosSemidefTri) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

function setup_data(cone::PosSemidefTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = similar(cone.point)
    cone.prev_scal_point = similar(cone.point)
    cone.prev_scal_dual_point = similar(cone.point)
    cone.new_scal_point = similar(cone.point)
    cone.grad = similar(cone.point)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.mat = zeros(R, cone.side, cone.side)
    cone.dual_mat = similar(cone.mat)
    cone.fact_mat = similar(cone.mat)
    cone.dual_fact_mat = similar(cone.mat)
    cone.work_mat = similar(cone.mat)
    cone.work_mat2 = similar(cone.mat)
    cone.work_mat3 = similar(cone.mat)
    # TODO initialize at the same time as the initial point as well as factorizations
    set_initial_point(cone.new_scal_point, cone)
    cone.scalmat_sqrt = Matrix{T}(I, cone.side, cone.side)
    cone.scalmat_sqrti = Matrix{T}(I, cone.side, cone.side)
    cone.correction = zeros(T, dim)
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
        _build_hess(cone.hess.data, cone.scalmat_sqrti' * cone.scalmat_sqrti, cone.rt2) # TODO fix inefficiency of the mul
    else
        _build_hess(cone.hess.data, cone.inv_mat, cone.rt2)
    end
    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::PosSemidefTri)
    @assert is_feas(cone)
    if cone.use_scaling
        _build_hess(cone.inv_hess.data, cone.scalmat_sqrt * cone.scalmat_sqrt', cone.rt2) # TODO fix inefficiency of the mul
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
        mul!(cone.work_mat3, cone.scalmat_sqrti', cone.scalmat_sqrti) # TODO fix inefficiency of the mul
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

# TODO don't need to special case here
function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTri)
    @assert is_feas(cone)
    if cone.use_scaling
        mul!(cone.work_mat3, cone.scalmat_sqrt, cone.scalmat_sqrt') # TODO fix inefficiency of the mul
        herm_congruence_prod!(prod, arr, cone.work_mat3, cone)
    else
        herm_congruence_prod!(prod, arr, cone.mat, cone)
    end
    return prod
end

function dist_to_bndry(cone::PosSemidefTri{T, R}, fact, dir::AbstractVector{T}) where {R <: RealOrComplex{T}} where {T <: Real}
    svec_to_smat!(cone.work_mat, dir, cone.rt2)
    copytri!(cone.work_mat, 'U', cone.is_complex)
    rdiv!(cone.work_mat, fact.U)
    ldiv!(fact.U', cone.work_mat)
    # TODO preallocate and explore faster options
    # NOTE julia calls eigvals inside eigmin, and eigmin is not currently implemented in GenericLinearAlgebra
    v = eigvals(Hermitian(cone.work_mat, :U))
    inv_min_dist = minimum(v)
    if inv_min_dist >= 0
        return T(Inf)
    else
        return -inv(inv_min_dist)
    end
end

# TODO refactor this better with dist_to_bndry
function step_max_dist(cone::PosSemidefTri, s_sol::AbstractVector, z_sol::AbstractVector)
    @assert cone.is_feas

    svec_to_smat!(cone.mat, cone.point, cone.rt2)
    copyto!(cone.fact_mat, cone.mat)
    cone.fact = cholesky!(Hermitian(cone.fact_mat, :U))
    primal_dist = dist_to_bndry(cone, cone.fact, s_sol)

    svec_to_smat!(cone.dual_mat, cone.dual_point, cone.rt2)
    copyto!(cone.dual_fact_mat, cone.dual_mat)
    cone.dual_fact = cholesky!(Hermitian(cone.dual_fact_mat, :U))
    dual_dist = dist_to_bndry(cone, cone.dual_fact, z_sol)

    return min(primal_dist, dual_dist)
end

# from MOSEK paper
# Pinv = inv(smat(point))
# smat correction = (Pinv * S * Z + Z * S * Pinv) / 2
# TODO cleanup
function correction(cone::PosSemidefTri, s_sol::AbstractVector, z_sol::AbstractVector)#, primal_point)
    @assert cone.grad_updated

    S = copytri!(svec_to_smat!(cone.work_mat, s_sol, cone.rt2), 'U', cone.is_complex)
    Z = Hermitian(svec_to_smat!(cone.work_mat2, z_sol, cone.rt2))

    # TODO compare the following numerically
    # Pinv_S_Z = mul!(cone.work_mat3, ldiv!(cone.fact, S), Z)
    # Pinv_S_Z = ldiv!(cone.fact, mul!(cone.work_mat3, S, Z))
    # TODO reuse factorization if useful
    # fact = cholesky(Hermitian(svec_to_smat!(cone.work_mat3, primal_point, cone.rt2), :U))
    fact = cholesky(Hermitian(svec_to_smat!(cone.work_mat3, cone.point, cone.rt2), :U))
    Pinv_S_Z = mul!(cone.work_mat3, ldiv!(fact, S), Z)

    Pinv_S_Z_symm = cone.work_mat
    @. Pinv_S_Z_symm = (Pinv_S_Z + Pinv_S_Z') / 2
    smat_to_svec!(cone.correction, Pinv_S_Z_symm, cone.rt2)

    return cone.correction
end

# s_sol and z_sol are scaled by an old scaling
function step_and_update_scaling(cone::PosSemidefTri{T, R}, s_sol::AbstractVector, z_sol::AbstractVector, step_size::T) where {R <: RealOrComplex{T}} where {T <: Real}
    if cone.try_scaled_updates
        # get the next s, z but in the old scaling
        # TODO handle old note by Lea - "we could get the next s, z but in the old scaling by dividing by sqrt(H(v)), which is cone-specific"
        dir = similar(cone.point)
        gen_congruence_prod!(dir, s_sol, cone.scalmat_sqrti', cone)
        @. cone.prev_scal_point = cone.new_scal_point + step_size * dir
        gen_congruence_prod!(dir, z_sol, cone.scalmat_sqrt, cone)
        @. cone.prev_scal_dual_point = cone.new_scal_point + step_size * dir

        # update old scaling
        svec_to_smat!(cone.mat, cone.prev_scal_point, cone.rt2)
        copyto!(cone.work_mat, cone.mat)
        fact = cholesky!(Hermitian(cone.work_mat, :U))

        svec_to_smat!(cone.dual_mat, cone.prev_scal_dual_point, cone.rt2)
        copyto!(cone.dual_fact_mat, cone.dual_mat)
        dual_fact = cholesky!(Hermitian(cone.dual_fact_mat, :U))

        (U, lambda, V) = svd(dual_fact.U * fact.L)
        # TODO fix the next few lines - use diagonal .diag
        # copyto!(cone.new_scal_point.diag, lambda)
        cone.new_scal_point .= 0
        k = 1
        incr = (cone.is_complex ? 2 : 1)
        @inbounds for i in 1:cone.side
            cone.new_scal_point[k] = lambda[i]
            k += incr * i + 1
        end

        cone.scalmat_sqrt = cone.scalmat_sqrt * fact.L * V * Diagonal(inv.(sqrt.(lambda)))
        cone.scalmat_sqrti = Diagonal(sqrt.(lambda)) * V' * (fact.L \ cone.scalmat_sqrti)
    else
        # calculate scaling without using old scaling
        svec_to_smat!(cone.dual_mat, cone.dual_point, cone.rt2)
        copyto!(cone.dual_fact_mat, cone.dual_mat)
        dual_fact = cone.dual_fact = cholesky!(Hermitian(cone.dual_fact_mat, :U), check = false)
        svec_to_smat!(cone.mat, cone.point, cone.rt2)
        fact = cholesky(Hermitian(cone.mat, :U))

        # TODO preallocate
        (U, lambda, V) = svd(dual_fact.U * fact.L)
        cone.scalmat_sqrt = fact.L * V * Diagonal(inv.(sqrt.(lambda)))
        cone.scalmat_sqrti = Diagonal(inv.(sqrt.(lambda))) * U' * dual_fact.U
    end

    return
end
