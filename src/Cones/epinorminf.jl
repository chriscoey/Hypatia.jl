#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of L-infinity norm
(u in R, w in R^n) : u >= norm_inf(w)

barrier from "Barrier Functions in Interior Point Methods" by Osman Guler
-sum_i(log(u - w_i^2/u)) - log(u)
= -sum_i(log(u^2 - w_i^2)) + (n - 1)log(u)
=#

mutable struct EpiNormInf{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    dim::Int
    n::Int
    is_complex::Bool
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    w::AbstractVector{R}
    den::AbstractVector{T}
    invden::Vector{T}
    uden::Vector{R}
    wden::Vector{R}
    edge::Vector{R}
    invedge::Vector{R}
    diag11::T
    schur::T

    function EpiNormInf{T, R}(
        dim::Int, # TODO maybe change to n (dim of the normed vector)
        is_dual::Bool,
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert dim >= 2
        cone = new{T, R}()
        cone.use_dual = is_dual
        cone.dim = dim # TODO
        cone.is_complex = (R <: Complex)
        cone.n = (cone.is_complex ? div(dim - 1, 2) : dim - 1)
        return cone
    end
end

EpiNormInf{T, R}(dim::Int) where {R <: RealOrComplex{T}} where {T <: Real} = EpiNormInf{T, R}(dim, false)

reset_data(cone::EpiNormInf) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_inv_hess_prod_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U) # TODO this is expensive to allocate. maybe use sparse. don't alloc if not using
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    n = cone.n
    if cone.is_complex
        cone.w = zeros(R, n)
    end
    cone.wden = zeros(R, n)
    cone.den = zeros(T, n)
    cone.invden = zeros(T, n)
    cone.uden = zeros(R, n)
    cone.edge = zeros(R, n)
    cone.invedge = zeros(R, n)
    return
end

get_nu(cone::EpiNormInf) = cone.n + 1

function set_initial_point(arr::AbstractVector{T}, cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

function update_feas(cone::EpiNormInf)
    @assert !cone.feas_updated
    u = cone.point[1]
    @views w = cone.point[2:end]

    if cone.is_complex
        w = rvec_to_cvec!(cone.w, w)
    end
    cone.is_feas = (u > 0 && u > norm(w, Inf))

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    if !cone.is_complex
        @views cone.w = cone.point[2:end]
    end
    w = cone.w

    usqr = abs2(u)
    @. cone.den = usqr - abs2(w)
    @. cone.uden = 2 * u / cone.den
    @. cone.wden = 2 * w / cone.den

    cone.grad[1] = (length(w) - 1) / u - sum(cone.uden)
    if cone.is_complex
        @views cvec_to_rvec!(cone.grad[2:end], cone.wden)
    else
        cone.grad[2:end] .= cone.wden
    end

    cone.grad_updated = true
    return cone.grad
end

# calculate edge, invden, invedge, diag11, schur
function hess_inv_hess_prod_updated(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert cone.grad_updated
    @assert !cone.hess_inv_hess_prod_updated
    u = cone.point[1]
    w = cone.w

    @. cone.edge = -cone.wden * cone.uden
    @. cone.invden = 2 / cone.den

    schur = zero(T)
    usqr = abs2(u)
    for (j, wj) in enumerate(w) # TODO inbounds
        u2pwj2 = usqr + abs2(wj)
        cone.invedge[j] = 2 * u / u2pwj2 * wj
        schur += 2 / u2pwj2
    end

    t1 = (length(w) - 1) / u
    cone.diag11 = sum(abs2, cone.uden) - (t1 + sum(cone.uden)) / u
    @assert cone.diag11 > 0
    cone.schur = schur - t1 / u
    @assert cone.schur > 0

    cone.hess_inv_hess_prod_updated = true
    return nothing
end

# symmetric arrow matrix
# TODO maybe make explicit H a sparse matrix
function update_hess(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    if !cone.hess_inv_hess_prod_updated
        hess_inv_hess_prod_updated(cone)
    end
    H = cone.hess.data
    u = cone.point[1]
    w = cone.w

    H[1, 1] = cone.diag11
    if cone.is_complex
        k = 2
        for (j, wj) in enumerate(w)
            ej = cone.edge[j]
            H[1, k] = real(ej)
            H[1, k + 1] = imag(ej)
            wdenj = cone.wden[j]
            invdenj = cone.invden[j]
            H[k, k] = abs2(real(wdenj)) + invdenj
            H[k, k + 1] = real(wdenj) * imag(wdenj)
            H[k + 1, k + 1] = abs2(imag(wdenj)) + invdenj
            k += 2
        end
    else
        k = 2
        for (j, wj) in enumerate(w)
            H[1, k] = cone.edge[j]
            H[k, k] = abs2(cone.wden[j]) + cone.invden[j]
            k += 1
        end
    end

    cone.hess_updated = true
    return cone.hess
end

# Diag(0, inv(diag)) + xx' / schur, where x = (-1, edge ./ diag)
function update_inv_hess(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    if !cone.hess_inv_hess_prod_updated
        hess_inv_hess_prod_updated(cone)
    end
    Hi = cone.inv_hess.data
    u = cone.point[1]
    w = cone.w

    Hi[1, 1] = 1
    if cone.is_complex
        @views cvec_to_rvec!(Hi[1, 2:end], cone.invedge)
    else
        @. Hi[1, 2:end] = cone.invedge
    end

    for j in 2:cone.dim, i in 2:j
        Hi[i, j] = Hi[1, j] * Hi[1, i]
    end
    Hi ./= cone.schur

    if cone.is_complex
        k = 2
        for (j, wj) in enumerate(w)
            wdenj = cone.wden[j]
            invdenj = cone.invden[j]
            H11 = abs2(real(wdenj)) + invdenj
            H22 = abs2(imag(wdenj)) + invdenj
            H12 = real(wdenj) * imag(wdenj)
            detj = H11 * H22 - abs2(H12)
            Hi[k, k] += H22 / detj
            Hi[k, k + 1] += -H12 / detj
            Hi[k + 1, k + 1] += H11 / detj
            k += 2
        end
    else
        for j in eachindex(cone.wden)
            Hi[j + 1, j + 1] += inv(abs2(cone.wden[j]) + cone.invden[j])
        end
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

# update_hess_prod(cone::EpiNormInf) = nothing
# update_inv_hess_prod(cone::EpiNormInf) = nothing
#
# # TODO uses edge, diag11, diag2n
# function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
#     @assert cone.grad_updated
#     @views begin
#         copyto!(prod[1, :], arr[1, :])
#         mul!(prod[1, :], arr[2:end, :]', cone.edge, true, cone.diag11)
#         mul!(prod[2:end, :], cone.edge, arr[1, :]')
#         @. prod[2:end, :] += cone.diag2n * arr[2:end, :]
#     end
#     return prod
# end
#
# # TODO uses invedge, schur, diag2n
# function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
#     @assert cone.grad_updated
#     @views begin
#         copyto!(prod[1, :], arr[1, :])
#         mul!(prod[1, :], arr[2:end, :]', cone.invedge, true, true)
#         @. prod[2:end, :] = cone.invedge * prod[1, :]'
#         prod ./= cone.schur
#         @. prod[2:end, :] += arr[2:end, :] / cone.diag2n
#     end
#     return prod
# end

hess_nz_count(cone::EpiNormInf, lower_only::Bool) = (lower_only ? 2 * cone.dim - 1 : 3 * cone.dim - 2)

# the row indices of nonzero elements in column j, inverse Hessian is fully dense (sum of a diagonal plus rank-one matrix)
function hess_nz_idxs_col(cone::EpiNormInf, j::Int, lower_only::Bool)
    if j == 1
        return 1:cone.dim
    elseif lower_only
        return j:j
    else
        return [1, j]
    end
end

# utilities for converting between real and complex vectors
function rvec_to_cvec!(cvec::AbstractVector{Complex{T}}, rvec::AbstractVector{T}) where {T}
    k = 1
    # @inbounds for i in eachindex(cvec)
    for i in eachindex(cvec)
        cvec[i] = Complex(rvec[k], rvec[k + 1])
        k += 2
    end
    return cvec
end

function cvec_to_rvec!(rvec::AbstractVector{T}, cvec::AbstractVector{Complex{T}}) where {T}
    k = 1
    # @inbounds for i in eachindex(cvec)
    for i in eachindex(cvec)
        ci = cvec[i]
        rvec[k] = real(ci)
        rvec[k + 1] = imag(ci)
        k += 2
    end
    return rvec
end
