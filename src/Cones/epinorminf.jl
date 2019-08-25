#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of L-infinity norm
(u in R, w in R^n) : u >= norm_inf(w)

barrier from "Barrier Functions in Interior Point Methods" by Osman Guler
-sum_i(log(u - w_i^2/u)) - log(u)
=#

mutable struct EpiNormInf{T <: Real} <: Cone{T}
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

    diag11::T
    diag2n::Vector{T}
    edge2n::Vector{T}
    div2n::Vector{T}
    schur::T

    function EpiNormInf{T}(dim::Int, is_dual::Bool) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

EpiNormInf{T}(dim::Int) where {T <: Real} = EpiNormInf{T}(dim, false)

reset_data(cone::EpiNormInf) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

# TODO maybe only allocate the fields we use
function setup_data(cone::EpiNormInf{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.diag2n = zeros(T, dim - 1)
    cone.edge2n = zeros(T, dim - 1)
    cone.div2n = zeros(T, dim - 1)
    return
end

get_nu(cone::EpiNormInf) = cone.dim

function set_initial_point(arr::AbstractVector, cone::EpiNormInf)
    arr .= 0
    arr[1] = 1
    return arr
end

function update_feas(cone::EpiNormInf)
    @assert !cone.feas_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    cone.is_feas = (u > 0 && u > norm(w, Inf))
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiNormInf{T}) where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    g1 = zero(u)
    h1 = zero(u)
    usqr = abs2(u)
    cone.schur = zero(T)
    @inbounds for (j, wj) in enumerate(w)
        # NOTE these operations are somewhat redundant, but numerically tuned to work well
        wjsqr = abs2(wj)
        usqrmwsqr = usqr - wjsqr
        @assert usqrmwsqr > 0
        iuw2u = 2 * u / usqrmwsqr
        g1 += iuw2u
        h1 += abs2(iuw2u)
        iu2w2 = 2 / usqrmwsqr
        iu2ww = wj * iu2w2
        cone.grad[j + 1] = iu2ww
        # NOTE diag2n and edge2n operations can be moved to hessian update
        cone.diag2n[j] = iu2w2 + abs2(iu2ww)
        cone.edge2n[j] = -2 / (u - wjsqr / u) * iu2ww
        # NOTE div2n and schur operations can be moved to inv hessian update
        usqrpwsqr = usqr + wjsqr
        cone.div2n[j] = 2 * u * wj / usqrpwsqr
        cone.schur += inv(usqrpwsqr)
    end
    t1 = (cone.dim - 2) / u
    cone.grad[1] = t1 - g1
    cone.diag11 = -(t1 + g1) / u + h1
    @assert cone.diag11 > 0
    cone.schur = 2 * cone.schur - t1 / u
    @assert cone.schur > 0
    cone.grad_updated = true
    return cone.grad
end

# symmetric arrow matrix
function update_hess(cone::EpiNormInf)
    @assert cone.grad_updated
    H = cone.hess.data
    H[1, 1] = cone.diag11
    @inbounds for j in 2:cone.dim
        H[1, j] = cone.edge2n[j - 1]
        H[j, j] = cone.diag2n[j - 1]
    end
    cone.hess_updated = true
    return cone.hess
end

# Diag(0, inv(diag)) + xx' / schur, where x = (-1, edge ./ diag)
function update_inv_hess(cone::EpiNormInf)
    @assert cone.grad_updated
    cone.inv_hess.data[1, 1] = 1
    @. cone.inv_hess.data[1, 2:end] = cone.div2n
    @inbounds for j in 2:cone.dim, i in 2:j
        cone.inv_hess.data[i, j] = cone.inv_hess.data[1, j] * cone.inv_hess.data[1, i]
    end
    cone.inv_hess.data ./= cone.schur
    @inbounds for j in 2:cone.dim
        cone.inv_hess.data[j, j] += inv(cone.diag2n[j - 1])
    end
    cone.inv_hess_updated = true
    return cone.inv_hess
end

update_hess_prod(cone::EpiNormInf) = nothing
update_inv_hess_prod(cone::EpiNormInf) = nothing

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    @assert cone.grad_updated
    @. @views prod[1, :] = cone.diag11 * arr[1, :]
    @views mul!(prod[1, :], arr[2:end, :]', cone.edge2n, true, true)
    @views mul!(prod[2:end, :], cone.edge2n, arr[1, :]')
    @. @views prod[2:end, :] += cone.diag2n * arr[2:end, :]
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    @assert cone.grad_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    @views copyto!(prod[1, :], arr[1, :])
    @views mul!(prod[1, :], arr[2:end, :]', cone.div2n, true, true)
    @. @views prod[2:end, :] = 2 * u * w * prod[1, :]'
    @. @views prod[2:end, :] /= (abs2(u) + abs2(w))
    prod ./= cone.schur
    @. @views prod[2:end, :] += arr[2:end, :] / cone.diag2n

    return prod
end
