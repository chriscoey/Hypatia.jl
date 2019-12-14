#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

hypograph of the root determinant of a (row-wise lower triangle) symmetric positive definite matrix
(u in R, W in S_n+) : u <= det(W)^(1/n)

SC barrier from correspondence with A. Nemirovski
-(5 / 3) ^ 2 * (log(det(W) ^ (1 / n) - u) + logdet(W))
=#

mutable struct HypoRootdetTri{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    side::Int
    point::Vector{T}
    rt2::T

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_prod_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    W::Matrix{T}
    work_mat::Matrix{T}
    fact_W
    Wi::Matrix{T}
    rootdet::T
    rootdetu::T
    frac::T
    # constants for Kronecker product and dot product components of the Hessian
    kron_const::T
    dot_const::T
    sc_const::T

    function HypoRootdetTri{T}(
        dim::Int,
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.rt2 = sqrt(T(2))
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

HypoRootdetTri{T}(dim::Int) where {T <: Real} = HypoRootdetTri{T}(dim, false)

reset_data(cone::HypoRootdetTri) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_prod_updated = cone.inv_hess_prod_updated = false)

function setup_data(cone::HypoRootdetTri{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.side = round(Int, sqrt(0.25 + 2 * (dim - 1)) - 0.5)
    cone.sc_const = T(25) / T(9)
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.W = zeros(T, cone.side, cone.side)
    cone.work_mat = zeros(T, cone.side, cone.side)
    return
end

get_nu(cone::HypoRootdetTri) = (cone.side + 1) * cone.sc_const

function set_initial_point(arr::AbstractVector{T}, cone::HypoRootdetTri{T}) where {T <: Real}
    arr .= 0
    side = cone.side
    const1 = sqrt(T(5side^2 + 2side + 1))
    const2 = arr[1] = -5 * sqrt((3side + 1 - const1) / T(side + 1)) / (3 * sqrt(T(2)))
    const3 = -const2 * (side + 1 + const1) / side / 2
    k = 2
    @inbounds for i in 1:cone.side
        arr[k] = const3
        k += i + 1
    end
    return arr
end

function update_feas(cone::HypoRootdetTri{T}) where {T}
    @assert !cone.feas_updated
    u = cone.point[1]

    svec_to_smat!(cone.W, view(cone.point, 2:cone.dim), cone.rt2)
    cone.fact_W = cholesky!(Symmetric(cone.W, :U), check = false) # mutates W, which isn't used anywhere else
    if isposdef(cone.fact_W)
        cone.rootdet = det(cone.fact_W) ^ inv(T(cone.side))
        cone.rootdetu = cone.rootdet - u
        cone.is_feas = (cone.rootdetu > 0)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::HypoRootdetTri)
    @assert cone.is_feas
    u = cone.point[1]

    cone.grad[1] = inv(cone.rootdetu)
    cone.Wi = inv(cone.fact_W) # TODO in-place
    @views smat_to_svec!(cone.grad[2:cone.dim], cone.Wi, cone.rt2)
    cone.frac = cone.rootdet / cone.side / cone.rootdetu
    @. @views cone.grad[2:cone.dim] *= -cone.frac - 1
    @. cone.grad *= cone.sc_const

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoRootdetTri)
    if !cone.hess_prod_updated
        update_hess_prod(cone) # fills in first row of the Hessian and calculates constants
    end
    Wi = cone.Wi
    kron_const = cone.kron_const
    dot_const = cone.dot_const
    H = cone.hess.data

    k1 = 2
    for i in 1:cone.side, j in 1:i
        k2 = 2
        @inbounds for i2 in 1:cone.side, j2 in 1:i2
            if (i == j) && (i2 == j2)
                H[k2, k1] = abs2(Wi[i2, i]) * kron_const + Wi[i, i] * Wi[i2, i2] * dot_const
            elseif (i != j) && (i2 != j2)
                H[k2, k1] = (Wi[i2, i] * Wi[j, j2] + Wi[j2, i] * Wi[j, i2]) * kron_const + 2 * Wi[i, j] * Wi[i2, j2] * dot_const
            else
                H[k2, k1] = cone.rt2 * (Wi[i2, i] * Wi[j, j2] * kron_const + Wi[i, j] * Wi[i2, j2] * dot_const)
            end
            if k2 == k1
                break
            end
            k2 += 1
        end
        k1 += 1
    end
    @. @views H[2:end, 2:end] *= cone.sc_const

    cone.hess_updated = true
    return cone.hess
end

# updates first row of the Hessian
function update_hess_prod(cone::HypoRootdetTri)
    @assert cone.grad_updated

    frac = cone.frac # rootdet / rootdetu / side
    # update constants used in the Hessian
    cone.kron_const = frac + 1
    cone.dot_const = abs2(frac) - frac / cone.side
    # update first row in the Hessian
    rootdetu = cone.rootdetu
    Wi = cone.Wi
    hess = cone.hess.data
    hess[1, 1] = cone.grad[1] / rootdetu
    @views smat_to_svec!(hess[1, 2:cone.dim], Wi, cone.rt2)
    @. hess[1, 2:end] *= -frac / rootdetu * cone.sc_const

    cone.hess_prod_updated = true
    return
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::HypoRootdetTri)
    if !cone.hess_prod_updated
        update_hess_prod(cone) # fills in first row of the Hessian and calculates constants
    end

    @views mul!(prod[1, :]', cone.hess[1, :]', arr)
    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.work_mat, view(arr, 2:cone.dim, i), cone.rt2)
        dot_prod = dot(Symmetric(cone.work_mat, :U), Symmetric(cone.Wi, :U))
        copytri!(cone.work_mat, 'U')
        rdiv!(cone.work_mat, cone.fact_W)
        ldiv!(cone.fact_W, cone.work_mat)
        axpby!(dot_prod * cone.dot_const, cone.Wi, cone.kron_const, cone.work_mat)
        @views smat_to_svec!(prod[2:cone.dim, i], cone.work_mat, cone.rt2)
    end
    @. @views prod[2:cone.dim, :] *= cone.sc_const
    @views mul!(prod[2:cone.dim, :], cone.hess[2:cone.dim, 1], arr[1, :]', true, true)

    return prod
end
