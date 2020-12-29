#=
(closure of) epigraph of sum of perspectives of entropies (AKA vector relative entropy cone)
(u in R, v in R_+^n, w in R_+^n) : u >= sum_i w_i*log(w_i/v_i) TODO update description here for non-contiguous v/w

barrier from "Primal-Dual Interior-Point Methods for Domain-Driven Formulations" by Karimi & Tuncel, 2019
-log(u - sum_i w_i*log(w_i/v_i)) - sum_i (log(v_i) + log(w_i))
=#

mutable struct EpiRelEntropy{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    w_dim::Int
    v_idxs
    w_idxs

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    correction::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_aux_updated::Bool
    inv_hess_sqrt_aux_updated::Bool
    use_inv_hess_sqrt::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, SparseMatrixCSC{T, Int}}

    lwv::Vector{T}
    tau::Vector{T}
    z::T
    Hiuu::T
    Hiuv::Vector{T}
    Hiuw::Vector{T}
    Hivv::Vector{T}
    Hivw::Vector{T}
    Hiww::Vector{T}
    rtiuu::T
    rtiuv::Vector{T}
    rtiuw::Vector{T}
    rtivv::Vector{T}
    rtivw::Vector{T}
    rtiww::Vector{T}
    temp1::Vector{T}
    temp2::Vector{T}

    function EpiRelEntropy{T}(
        dim::Int;
        use_dual::Bool = false,
        ) where {T <: Real}
        @assert dim >= 3
        @assert isodd(dim)
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.w_dim = div(dim - 1, 2)
        cone.v_idxs = 2:2:dim
        cone.w_idxs = 3:2:dim
        return cone
    end
end

use_heuristic_neighborhood(cone::EpiRelEntropy) = false

reset_data(cone::EpiRelEntropy) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.inv_hess_aux_updated = cone.inv_hess_sqrt_aux_updated = false)

function use_sqrt_oracles(cone::EpiRelEntropy)
    cone.use_inv_hess_sqrt || return false
    cone.inv_hess_sqrt_aux_updated || update_inv_hess_sqrt_aux(cone)
    !cone.use_inv_hess_sqrt
    return cone.use_inv_hess_sqrt
end

# TODO only allocate the fields we use
function setup_extra_data(cone::EpiRelEntropy{T}) where {T <: Real}
    w_dim = cone.w_dim
    cone.lwv = zeros(T, w_dim)
    cone.tau = zeros(T, w_dim)
    cone.Hiuv = zeros(T, w_dim)
    cone.Hiuw = zeros(T, w_dim)
    cone.Hivv = zeros(T, w_dim)
    cone.Hivw = zeros(T, w_dim)
    cone.Hiww = zeros(T, w_dim)
    cone.rtiuv = zeros(T, w_dim)
    cone.rtiuw = zeros(T, w_dim)
    cone.rtivv = zeros(T, w_dim)
    cone.rtivw = zeros(T, w_dim)
    cone.rtiww = zeros(T, w_dim)
    cone.temp1 = zeros(T, w_dim)
    cone.temp2 = zeros(T, w_dim)
    cone.use_inv_hess_sqrt = true
    return cone
end

get_nu(cone::EpiRelEntropy) = cone.dim

function set_initial_point(arr::AbstractVector, cone::EpiRelEntropy)
    (arr[1], v, w) = get_central_ray_epirelentropy(cone.w_dim)
    @views arr[cone.v_idxs] .= v
    @views arr[cone.w_idxs] .= w
    return arr
end

function update_feas(cone::EpiRelEntropy{T}) where T
    @assert !cone.feas_updated
    u = cone.point[1]
    @views v = cone.point[cone.v_idxs]
    @views w = cone.point[cone.w_idxs]

    if all(vi -> vi > eps(T), v) && all(wi -> wi > eps(T), w)
        @. cone.lwv = log(w / v)
        cone.z = u - dot(w, cone.lwv)
        cone.is_feas = (cone.z > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiRelEntropy{T}) where T
    u = cone.dual_point[1]
    @views v = cone.dual_point[cone.v_idxs]
    @views w = cone.dual_point[cone.w_idxs]

    if all(vi -> vi > eps(T), v) && u > eps(T)
        return all(u * (1 + log(vi / u)) + wi > eps(T) for (vi, wi) in zip(v, w))
    end

    return false
end

function update_grad(cone::EpiRelEntropy)
    @assert cone.is_feas
    u = cone.point[1]
    @views v = cone.point[cone.v_idxs]
    @views w = cone.point[cone.w_idxs]
    z = cone.z
    g = cone.grad
    tau = cone.tau

    @. tau = (cone.lwv + 1) / -z
    g[1] = -inv(z)
    @. @views g[cone.v_idxs] = -(w / z + 1) / v
    @. @views g[cone.w_idxs] = -inv(w) - tau

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiRelEntropy{T}) where T
    @assert cone.grad_updated
    if !isdefined(cone, :hess)
        cone.hess = Symmetric(zeros(T, cone.dim, cone.dim), :U)
    end
    u = cone.point[1]
    v_idxs = cone.v_idxs
    w_idxs = cone.w_idxs
    point = cone.point
    @views v = point[v_idxs]
    @views w = point[w_idxs]
    tau = cone.tau
    z = cone.z
    sigma = cone.temp1
    g = cone.grad
    H = cone.hess.data

    # H_u_u, H_u_v, H_u_w parts
    H[1, 1] = abs2(cone.grad[1])
    @. sigma = w / v / z
    @. @views H[1, v_idxs] = sigma / z
    @. @views H[1, w_idxs] = tau / z

    # H_v_v, H_v_w, H_w_w parts
    zi = inv(z)
    @inbounds for (i, v_idx, w_idx) in zip(1:cone.w_dim, v_idxs, w_idxs)
        vi = point[v_idx]
        wi = point[w_idx]
        taui = tau[i]
        sigmai = sigma[i]

        H[v_idx, v_idx] = abs2(sigmai) - g[v_idx] / vi
        H[w_idx, w_idx] = abs2(taui) + (zi + inv(wi)) / wi

        @. H[v_idx, w_idxs] = sigmai * tau
        @. H[w_idx, v_idxs] = sigma * taui
        H[v_idx, w_idx] -= zi / vi

        @inbounds for j in (i + 1):cone.w_dim
            H[v_idx, v_idxs[j]] = sigmai * sigma[j]
            H[w_idx, w_idxs[j]] = taui * tau[j]
        end
    end

    cone.hess_updated = true
    return cone.hess
end

# auxiliary calculations for inverse Hessian
function update_inv_hess_aux(cone::EpiRelEntropy{T}) where T
    @assert !cone.inv_hess_aux_updated
    point = cone.point
    @views v = point[cone.v_idxs]
    @views w = point[cone.w_idxs]
    z = cone.z

    HiuHu = zero(T)
    @inbounds for i in 1:cone.w_dim
        wi = w[i]
        vi = v[i]
        lwvi = cone.lwv[i]
        zwi = z + wi
        z2wi = zwi + wi
        wz2wi = wi / z2wi
        vz2wi = vi / z2wi
        uvvi = wi * (wi * lwvi - z)
        uwwi = wi * (z + lwvi * zwi) * wz2wi
        HiuHu += wz2wi * uvvi - uwwi * (lwvi + 1)
        cone.Hiuv[i] = vz2wi * uvvi
        cone.Hiuw[i] = uwwi
        cone.Hivw[i] = wi * vi * wz2wi
        cone.Hiww[i] = wi * zwi * wz2wi
        cone.Hivv[i] = vi * zwi * vz2wi
    end
    cone.Hiuu = abs2(z) - HiuHu
    if cone.Hiuu < zero(T)
        @warn("bad Hiuu $(cone.Hiuu)")
    end

    cone.inv_hess_aux_updated = true
    return
end

# updates for nonzero values in the inverse Hessian
function update_inv_hess(cone::EpiRelEntropy{T}) where T
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)

    if !isdefined(cone, :inv_hess)
        # initialize sparse idxs for upper triangle of inverse Hessian
        cone.inv_hess = Symmetric(sparse_arrow_block2(T, cone.w_dim), :U)
    end

    # modify nonzeros of upper triangle of inverse Hessian
    nzval = cone.inv_hess.data.nzval
    nzval[1] = cone.Hiuu
    nz_idx = 1
    @inbounds for i in 1:cone.w_dim
        @. nzval[nz_idx .+ (1:5)] = (cone.Hiuv[i], cone.Hivv[i], cone.Hiuw[i], cone.Hivw[i], cone.Hiww[i])
        nz_idx += 5
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiRelEntropy)
    @assert cone.grad_updated
    v_idxs = cone.v_idxs
    w_idxs = cone.w_idxs
    @views v = cone.point[v_idxs]
    @views w = cone.point[w_idxs]
    z = cone.z
    sigma = cone.temp1
    tau = cone.tau
    @. sigma = w / v / z # TODO update/cache

    @inbounds @views begin
        u_arr = arr[1, :]
        mul!(prod[1, :], arr[v_idxs, :]', sigma)
        mul!(prod[1, :], arr[w_idxs, :]', tau, true, true)
        @. prod[1, :] += u_arr / z
        mul!(prod[v_idxs, :], sigma, u_arr')
        mul!(prod[w_idxs, :], tau, u_arr')
        @. prod /= z
    end

    @inbounds @views for j in 1:size(prod, 2)
        vj = arr[v_idxs, j]
        wj = arr[w_idxs, j]
        dotprods = dot(sigma, vj) + dot(tau, wj)
        @. prod[v_idxs, j] += sigma * dotprods + (vj * sigma + vj / v) / v - wj / v / z
        @. prod[w_idxs, j] += tau * dotprods + (wj / z + wj / w) / w - vj / v / z
    end

    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiRelEntropy)
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    return arrow_prod(prod, arr, cone.Hiuu, cone.Hiuv, cone.Hiuw, cone.Hivv, cone.Hivw, cone.Hiww)
end

function update_inv_hess_sqrt_aux(cone::EpiRelEntropy)
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    @assert !cone.inv_hess_sqrt_aux_updated
    cone.rtiuu = arrow_sqrt(cone.Hiuu, cone.Hiuv, cone.Hiuw, cone.Hivv, cone.Hivw, cone.Hiww, cone.rtiuv, cone.rtiuw, cone.rtivv, cone.rtivw, cone.rtiww)
    cone.use_inv_hess_sqrt = !iszero(cone.rtiuu)
    cone.inv_hess_sqrt_aux_updated = true
    return
end

function hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiRelEntropy)
    @assert cone.inv_hess_sqrt_aux_updated && cone.use_inv_hess_sqrt
    return inv_arrow_sqrt_prod(prod, arr, cone.rtiuu, cone.rtiuv, cone.rtiuw, cone.rtivv, cone.rtivw, cone.rtiww)
end

function inv_hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiRelEntropy)
    @assert cone.inv_hess_sqrt_aux_updated && cone.use_inv_hess_sqrt
    return arrow_sqrt_prod(prod, arr, cone.rtiuu, cone.rtiuv, cone.rtiuw, cone.rtivv, cone.rtivw, cone.rtiww)
end

function correction(cone::EpiRelEntropy, primal_dir::AbstractVector)
    @assert cone.grad_updated
    tau = cone.tau
    z = cone.z
    v_idxs = cone.v_idxs
    w_idxs = cone.w_idxs
    @views v = cone.point[v_idxs]
    @views w = cone.point[w_idxs]
    u_dir = primal_dir[1]
    @views v_dir = primal_dir[v_idxs]
    @views w_dir = primal_dir[w_idxs]
    corr = cone.correction
    @views v_corr = corr[v_idxs]
    @views w_corr = corr[w_idxs]
    wdw = cone.temp1
    vdv = cone.temp2

    i2z = inv(2 * z)
    @. wdw = w_dir / w
    @. vdv = v_dir / v
    const0 = (u_dir + dot(w, vdv)) / z + dot(tau, w_dir)
    const1 = abs2(const0) + sum(w[i] * abs2(vdv[i]) + w_dir[i] * (wdw[i] - 2 * vdv[i]) for i in eachindex(w)) / (2 * z)
    corr[1] = const1 / z

    # v
    v_corr .= const1
    @. v_corr += (const0 + vdv) * vdv - i2z * wdw * w_dir
    @. v_corr *= w
    @. v_corr += (z * vdv - w_dir) * vdv + (-const0 + i2z * w_dir) * w_dir
    @. v_corr /= v
    @. v_corr /= z

    # w
    @. w_corr = const1 * tau
    @. w_corr += ((const0 - w * vdv / z) / z + (inv(w) + i2z) * wdw) * wdw
    @. w_corr += (-const0 + w_dir / z - vdv / 2) / z * vdv

    return corr
end

# TODO remove this in favor of new hess_nz_count etc functions that directly use uu, uw, ww etc
inv_hess_nz_count(cone::EpiRelEntropy) = 3 * cone.dim - 2 + 2 * cone.w_dim
inv_hess_nz_count_tril(cone::EpiRelEntropy) = 2 * cone.dim - 1 + cone.w_dim
inv_hess_nz_idxs_col(cone::EpiRelEntropy, j::Int) = (j == 1 ? (1:cone.dim) : (iseven(j) ? [1, j, j + 1] : [1, j - 1, j]))
inv_hess_nz_idxs_col_tril(cone::EpiRelEntropy, j::Int) = (j == 1 ? (1:cone.dim) : (iseven(j) ? [j, j + 1] : [j]))

# see analysis in https://github.com/lkapelevich/HypatiaSupplements.jl/tree/master/centralpoints
function get_central_ray_epirelentropy(w_dim::Int)
    if w_dim <= 10
        return central_rays_epirelentropy[w_dim, :]
    end
    # use nonlinear fit for higher dimensions
    rtwdim = sqrt(w_dim)
    if w_dim <= 20
        u = 1.2023 / rtwdim - 0.015
        v = 0.432 / rtwdim + 1.0125
        w = -0.3057 / rtwdim + 0.972
    else
        u = 1.1513 / rtwdim - 0.0069
        v = 0.4873 / rtwdim + 1.0008
        w = -0.4247 / rtwdim + 0.9961
    end
    return [u, v, w]
end

const central_rays_epirelentropy = [
    0.827838399	1.290927714	0.805102005;
    0.708612491	1.256859155	0.818070438;
    0.622618845	1.231401008	0.829317079;
    0.558111266	1.211710888	0.838978357;
    0.508038611	1.196018952	0.847300431;
    0.468039614	1.183194753	0.854521307;
    0.435316653	1.172492397	0.860840992;
    0.408009282	1.163403374	0.866420017;
    0.38483862	1.155570329	0.871385499;
    0.364899122	1.148735192	0.875838068;
    ]
