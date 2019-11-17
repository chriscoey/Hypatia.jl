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
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    diag11::T
    diag2n::Vector{R}
    invdiag2n::Vector{R}
    edge2n::Vector{R}
    div2n::Vector{R}
    schur::T
    w::Vector{R}

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

reset_data(cone::EpiNormInf) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

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
    cone.diag2n = zeros(R, n)
    cone.invdiag2n = zeros(R, n)
    cone.edge2n = zeros(R, n)
    cone.div2n = zeros(R, n)
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
    if cone.is_complex
        w = cone.w
    else
        @views w = cone.point[2:end]
    end

    g1 = zero(T)
    h1 = zero(T)
    usqr = abs2(u)
    cone.schur = zero(T)

    complexgrad = similar(w)
    prods = one(T)
    sums = zero(T)

    # @inbounds for (j, wj) in enumerate(w)
    for (j, wj) in enumerate(w)
        # umwj = (u - wj)
        # upwj = (u + wj')
        # udiv = 2 * u / umwj / upwj
        u2wj2 = usqr - abs2(wj)
        udiv = 2 * u / u2wj2
        g1 += udiv
        h1 += abs2(udiv)
        wdiv = 2 * wj / u2wj2
        complexgrad[j] = wdiv
        # cone.grad[j + 1] = wdiv
        # cone.diag2n[j] = 2 * (1 + wj * wdiv) / umwj / upwj
        # cone.invdiag2n[j] = umwj * upwj / (2 + 2 * wj * wdiv)
        # cone.edge2n[j] = -udiv * wdiv
        u2pwj2 = usqr + abs2(wj)
        cone.div2n[j] = 2 * u / u2pwj2 * wj
        cone.schur += inv(u2pwj2)
        sums += abs2(wj)
        prods *= abs2(wj)
    end

    t1 = (length(w) - 1) / u
    cone.grad[1] = t1 - g1
    @views cvec_to_rvec!(cone.grad[2:end], complexgrad)
    cone.diag11 = h1 - (t1 + g1) / u
    @assert cone.diag11 > 0
    cone.schur = 2 * cone.schur - t1 / u
    @assert cone.schur > 0

    cone.grad_updated = true
    return cone.grad
end

# symmetric arrow matrix
# TODO maybe make explicit H a sparse matrix
function update_hess(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert cone.grad_updated
    H = cone.hess.data
    u = cone.point[1]
    w = cone.w

    H[1, 1] = cone.diag11
    if cone.is_complex
        k = 2
        for (j, wj) in enumerate(w)
            den = (abs2(u) - abs2(wj)) / 2
            wjden = wj / den
            ej = -u / den * wjden
            invden = inv(den)

            H[1, k] = real(ej)
            H[1, k + 1] = imag(ej)

            H[k, k] = abs2(real(wjden)) + invden
            H[k, k + 1] = real(wjden) * imag(wjden)
            H[k + 1, k + 1] = abs2(imag(wjden)) + invden

            k += 2
        end
    else
        @inbounds for j in 2:cone.dim
            H[1, j] = cone.edge2n[j - 1]
            H[j, j] = cone.diag2n[j - 1]
        end
    end

    cone.hess_updated = true
    return cone.hess
end

# Diag(0, inv(diag)) + xx' / schur, where x = (-1, edge ./ diag)
function update_inv_hess(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert cone.grad_updated
    Hi = cone.inv_hess.data
    u = cone.point[1]
    w = cone.w

    if cone.is_complex
        # Hi[1, 1] = inv(cone.schur)
        #
        # k = 2
        # for (j, wj) in enumerate(w)
        #     den = (abs2(u) - abs2(wj)) / 2
        #     wjden = wj / den
        #     ej = -u / den * wjden
        #     invden = inv(den)
        #
        #     H[1, k] = real(ej)
        #     H[1, k + 1] = imag(ej)
        #
        #     H[k, k] = abs2(real(wjden)) + invden
        #     H[k, k + 1] = real(wjden) * imag(wjden)
        #     H[k + 1, k + 1] = abs2(imag(wjden)) + invden
        #
        #     k += 2
        # end

        Hi[1, 1] = 1
        @views cvec_to_rvec!(Hi[1, 2:end], cone.div2n)
        # @. Hi[1, 2:end] = cone.div2n
        @inbounds for j in 2:cone.dim, i in 2:j
            Hi[i, j] = Hi[1, j] * Hi[1, i]
        end
        Hi ./= cone.schur

        k = 2
        @inbounds for j in eachindex(w)
            invkk = inv(cholesky!(Symmetric(cone.hess.data[k:(k + 1), k:(k + 1)], :U)))
            Hi[k:(k + 1), k:(k + 1)] += invkk
            k += 2
        end



    else
        cone.inv_hess.data[1, 1] = 1
        @. cone.inv_hess.data[1, 2:end] = cone.div2n
        @inbounds for j in 2:cone.dim, i in 2:j
            cone.inv_hess.data[i, j] = cone.inv_hess.data[1, j] * cone.inv_hess.data[1, i]
        end
        cone.inv_hess.data ./= cone.schur
        @inbounds for j in 2:cone.dim
            cone.inv_hess.data[j, j] += cone.invdiag2n[j - 1]
        end
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end
#
# update_hess_prod(cone::EpiNormInf) = nothing
# update_inv_hess_prod(cone::EpiNormInf) = nothing
#
# function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
#     @assert cone.grad_updated
#     @views begin
#         copyto!(prod[1, :], arr[1, :])
#         mul!(prod[1, :], arr[2:end, :]', cone.edge2n, true, cone.diag11)
#         mul!(prod[2:end, :], cone.edge2n, arr[1, :]')
#         @. prod[2:end, :] += cone.diag2n * arr[2:end, :]
#     end
#     return prod
# end
#
# function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
#     @assert cone.grad_updated
#     @views begin
#         copyto!(prod[1, :], arr[1, :])
#         mul!(prod[1, :], arr[2:end, :]', cone.div2n, true, true)
#         @. prod[2:end, :] = cone.div2n * prod[1, :]'
#         prod ./= cone.schur
#         @. prod[2:end, :] += arr[2:end, :] * cone.invdiag2n
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
