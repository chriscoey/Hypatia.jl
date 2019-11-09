#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

(closure of) hypograph of perspective of (natural) log of determinant of a (row-wise lower triangle) symmetric positive define matrix
(u in R, v in R_+, w in S_+) : u <= v*logdet(W/v)
(see equivalent MathOptInterface LogDetConeConeTriangle definition)

barrier (guessed, based on analogy to 3D exponential cone barrier)
-log(v*logdet(W/v) - u) - logdet(W) - log(v)
=#

mutable struct HypoPerLogdetTri{T <: Real} <: Cone{T}
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

    mat::Matrix{T}
    mat2::Matrix{T}
    fact_mat
    ldWv::T
    z::T
    Wi::Matrix{T}
    nLz::T
    ldWvuv::T
    vzip1::T
    Wivzi::Matrix{T}

    function HypoPerLogdetTri{T}(
        dim::Int,
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.rt2 = sqrt(T(2))
        cone.side = round(Int, sqrt(0.25 + 2 * (dim - 2)) - 0.5)
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

HypoPerLogdetTri{T}(dim::Int) where {T <: Real} = HypoPerLogdetTri{T}(dim, false)

reset_data(cone::HypoPerLogdetTri) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_prod_updated = cone.inv_hess_prod_updated = false)

function setup_data(cone::HypoPerLogdetTri{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.mat = Matrix{T}(undef, cone.side, cone.side)
    cone.mat2 = similar(cone.mat)
    cone.Wivzi = similar(cone.mat)
    return
end

get_nu(cone::HypoPerLogdetTri) = cone.side + 2

function set_initial_point(arr::AbstractVector, cone::HypoPerLogdetTri{T}) where {T <: Real}
    arr .= 0
    (arr[1], arr[2], w) = get_central_ray_hypoperlogdettri(cone.side)
    k = 3
    @inbounds for i in 1:cone.side
        arr[k] = w
        k += i + 1
    end
    return arr
end

function update_feas(cone::HypoPerLogdetTri)
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]

    if v > 0
        svec_to_smat!(cone.mat, view(cone.point, 3:cone.dim), cone.rt2)
        cone.fact_mat = cholesky!(Symmetric(cone.mat, :U), check = false)
        if isposdef(cone.fact_mat)
            cone.ldWv = logdet(cone.fact_mat) - cone.side * log(v)
            cone.z = v * cone.ldWv - u
            cone.is_feas = (cone.z > 0)
        else
            cone.is_feas = false
        end
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::HypoPerLogdetTri)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]

    cone.Wi = inv(cone.fact_mat)
    cone.nLz = (cone.side - cone.ldWv) / cone.z
    cone.ldWvuv = cone.ldWv - u / v
    cone.vzip1 = 1 + inv(cone.ldWvuv)
    cone.grad[1] = inv(cone.z)
    cone.grad[2] = cone.nLz - inv(v)
    gend = view(cone.grad, 3:cone.dim)
    smat_to_svec!(gend, cone.Wi, cone.rt2)
    gend .*= -cone.vzip1

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoPerLogdetTri)
    if !cone.hess_prod_updated
        update_hess_prod(cone) # fill in first two rows of the Hessian and compute Wivzi
    end
    Wi = cone.Wi
    Wivzi = cone.Wivzi
    rt2 = cone.rt2

    H = cone.hess.data
    k1 = 3
    for i in 1:cone.side, j in 1:i
        k2 = 3
        @inbounds for i2 in 1:cone.side, j2 in 1:i2
            if (i == j) && (i2 == j2)
                H[k2, k1] = abs2(Wi[i2, i]) * cone.vzip1 + Wivzi[i, i] * Wivzi[i2, i2]
            elseif (i != j) && (i2 != j2)
                H[k2, k1] = (Wi[i2, i] * Wi[j, j2] + Wi[j2, i] * Wi[j, i2]) * cone.vzip1 + 2 * Wivzi[i, j] * Wivzi[i2, j2]
            else
                H[k2, k1] = rt2 * (Wi[i2, i] * Wi[j, j2] * cone.vzip1 + Wivzi[i, j] * Wivzi[i2, j2])
            end
            if k2 == k1
                break
            end
            k2 += 1
        end
        k1 += 1
    end

    cone.hess_updated = true
    return cone.hess
end

# updates first two rows of the Hessian
function update_hess_prod(cone::HypoPerLogdetTri)
    @assert cone.grad_updated
    u = cone.point[1]
    v = cone.point[2]
    z = cone.z
    H = cone.hess.data

    @. cone.Wivzi = cone.Wi / cone.ldWvuv
    H[1, 1] = inv(z) / z
    H[1, 2] = cone.nLz / z
    h1end = view(H, 1, 3:cone.dim)
    smat_to_svec!(h1end, cone.Wivzi, cone.rt2)
    h1end ./= -z
    H[2, 2] = abs2(cone.nLz) + (cone.side / z + inv(v)) / v
    h2end = view(H, 2, 3:cone.dim)
    smat_to_svec!(h2end, cone.Wi, cone.rt2)
    h2end .*= ((cone.ldWv - cone.side) / cone.ldWvuv - 1) / z

    cone.hess_prod_updated = true
    return
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::HypoPerLogdetTri)
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end

    @views mul!(prod[1:2, :], cone.hess[1:2, :], arr)
    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat2, view(arr, 3:cone.dim, i), cone.rt2)
        dot_prod = dot(Symmetric(cone.mat2, :U), Symmetric(cone.Wivzi, :U))
        copytri!(cone.mat2, 'U')
        rdiv!(cone.mat2, cone.fact_mat)
        ldiv!(cone.fact_mat, cone.mat2)
        axpby!(dot_prod, cone.Wivzi, cone.vzip1, cone.mat2)
        smat_to_svec!(view(prod, 3:cone.dim, i), cone.mat2, cone.rt2)
    end
    @views mul!(prod[3:cone.dim, :], cone.hess[3:cone.dim, 1:2], arr[1:2, :], true, true)

    return prod
end

# see analysis in https://github.com/lkapelevich/HypatiaBenchmarks.jl/tree/master/centralpoints
function get_central_ray_hypoperlogdettri(Wside::Int)
    if Wside <= 5
        # lookup points where x = f'(x)
        return central_rays_hypoperlogdettri[Wside, :]
    end
    # use nonlinear fit for higher dimensions
    if Wside <= 16
        u = -2.070906 / Wside - 0.052713
        v = 0.420764 / Wside + 0.553790
        w = 0.629959 / Wside + 1.011841
    else
        u = -2.878002 / Wside - 0.001136
        v = 0.410904 / Wside + 0.553842
        w = 0.805068 / Wside + 1.000288
    end
    return [u, v, w]
end

const central_rays_hypoperlogdettri = [
    -0.827838399  0.805102005  1.290927713;
    -0.689609381  0.724604185  1.224619879;
    -0.584372734  0.681280549  1.182421998;
    -0.503500819  0.654485416  1.153054181;
    -0.440285901  0.636444221  1.131466932;
    ]
