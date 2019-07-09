#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of L-infinity norm
(u in R, w in R^n) : u >= norm_inf(w)

barrier from "Barrier Functions in Interior Point Methods" by Osman Guler
-sum_i(log(u - w_i^2/u)) - log(u)
=#

mutable struct EpiNormInf{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    point::AbstractVector{T}

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

    function EpiNormInf{T}(dim::Int, is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

EpiNormInf{T}(dim::Int) where {T <: HypReal} = EpiNormInf{T}(dim, false)

reset_data(cone::EpiNormInf) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

# TODO maybe only allocate the fields we use
function setup_data(cone::EpiNormInf{T}) where {T <: HypReal}
    reset_data(cone)
    dim = cone.dim
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
    cone.is_feas = cone.point[1] > 0 && cone.point[1] > maximum(abs, view(cone.point, 2:cone.dim))
    cone.feas_updated = true
    return cone.is_feas
end

# TODO maybe move the diag2n, edge2n, div2n to update hess/inv_hess functions
function update_grad(cone::EpiNormInf{T}) where {T <: HypReal}
    @assert cone.is_feas
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    g1 = zero(u)
    h1 = zero(u)
    usqr = abs2(u)
    for (j, wj) in enumerate(w)
        iuw2u = 2 / (u - abs2(wj) / u)
        g1 += iuw2u
        h1 += abs2(iuw2u)
        iu2w2 = 2 / (usqr - abs2(wj))
        iu2ww = wj * iu2w2
        cone.grad[j + 1] = iu2ww
        cone.diag2n[j] = iu2w2 + abs2(iu2ww)
        cone.edge2n[j] = -iuw2u * iu2ww
        cone.div2n[j] = -cone.edge2n[j] / cone.diag2n[j]
    end
    t1 = (cone.dim - 2) / u
    cone.grad[1] = t1 - g1
    cone.diag11 = -(t1 + g1) / u + h1
    cone.schur = cone.diag11 + dot(cone.edge2n, cone.div2n)
    cone.grad_updated = true
    return cone.grad
end

# symmetric arrow matrix
function update_hess(cone::EpiNormInf)
    @assert cone.grad_updated
    cone.hess.data[1, 1] = cone.diag11
    for j in 2:cone.dim
        cone.hess.data[1, j] = cone.edge2n[j - 1]
        cone.hess.data[j, j] = cone.diag2n[j - 1]
    end
    return cone.hess
end

# Diag(0, inv(diag)) + xx' / schur, where x = (-1, edge ./ diag)
function update_inv_hess(cone::EpiNormInf)
    @assert cone.grad_updated
    cone.inv_hess.data[1, 1] = 1
    @. cone.inv_hess.data[1, 2:end] = cone.div2n
    for j in 2:cone.dim, i in 2:j
        cone.inv_hess.data[i, j] = cone.inv_hess.data[1, j] * cone.inv_hess.data[1, i]
    end
    cone.inv_hess.data ./= cone.schur
    for j in 2:cone.dim
        cone.inv_hess.data[j, j] += inv(cone.diag2n[j - 1])
    end
    return cone.inv_hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    @assert cone.grad_updated
    for j in 1:size(prod, 2)
        @views prod[1, j] = cone.diag11 * arr[1, j] + dot(cone.edge2n, arr[2:end, j])
        @views @. prod[2:end, j] = cone.edge2n * arr[1, j] + cone.diag2n * arr[2:end, j]
    end
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    @assert cone.grad_updated
    for j in 1:size(prod, 2)
        @views prod[1, j] = arr[1, j] + dot(cone.div2n, arr[2:end, j])
        @. prod[2:end, j] = cone.div2n * prod[1, j]
    end
    prod ./= cone.schur
    @. @views prod[2:end, :] += arr[2:end, :] / cone.diag2n
    return prod
end
