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
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, SparseMatrixCSC{T, Int}}
    inv_hess::Symmetric{T, Matrix{T}}

    w::AbstractVector{R}
    den::AbstractVector{T}
    uden::Vector{R}
    wden::Vector{R}
    diag::Vector{T}
    offdiag::Vector{T}
    detdiag::Vector{T}
    edge::Vector{T}
    invedge::Vector{T}
    edgeR::Vector{R}
    invedgeR::Vector{R}
    diag11::T
    schur::T
    rtdiag::Vector{T}

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

reset_data(cone::EpiNormInf) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_inv_hess_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    n = cone.n
    cone.w = zeros(R, n)
    cone.wden = zeros(R, n)
    cone.den = zeros(T, n)
    cone.uden = zeros(R, n)
    cone.diag = zeros(T, dim - 1)
    cone.edge = zeros(T, dim - 1)
    cone.invedge = zeros(T, dim - 1)
    cone.rtdiag = zeros(T, dim - 1)
    if cone.is_complex
        cone.offdiag = zeros(T, n)
        cone.detdiag = zeros(T, n)
        cone.edgeR = zeros(R, n)
        cone.invedgeR = zeros(R, n)
    end
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
    @views vec_copy_to!(cone.w, cone.point[2:end])

    cone.is_feas = (u > 0 && u > norm(cone.w, Inf))

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    w = cone.w

    usqr = abs2(u)
    @. cone.den = usqr - abs2(w)
    @. cone.uden = 2 * u / cone.den
    @. cone.wden = 2 * w / cone.den

    cone.grad[1] = (length(w) - 1) / u - sum(cone.uden)
    @views vec_copy_to!(cone.grad[2:end], cone.wden)

    cone.grad_updated = true
    return cone.grad
end

# calculate edge, invedge, diag11, diag, offdiag, schur
function update_hess_inv_hess(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert cone.grad_updated
    u = cone.point[1]
    w = cone.w
    (edge, invedge) = (cone.is_complex ? (cone.edgeR, cone.invedgeR) : (cone.edge, cone.invedge))

    @. edge = -cone.wden * cone.uden

    schur = zero(T)
    usqr = abs2(u)
    @inbounds for (j, wj) in enumerate(w)
        wdenj = cone.wden[j]
        invdenj = 2 / cone.den[j]

        if cone.is_complex
            d11 = cone.diag[2j - 1] = abs2(real(wdenj)) + invdenj
            d22 = cone.diag[2j] = abs2(imag(wdenj)) + invdenj
            d12 = cone.offdiag[j] = real(wdenj) * imag(wdenj)
            cone.detdiag[j] = d11 * d22 - abs2(d12)
        else
            cone.diag[j] = abs2(wdenj) + invdenj
        end

        u2pwj2 = usqr + abs2(wj)
        invedge[j] = 2 * u / u2pwj2 * wj
        schur += 2 / u2pwj2
    end

    t1 = (length(w) - 1) / u
    cone.diag11 = sum(abs2, cone.uden) - (t1 + sum(cone.uden)) / u
    @assert cone.diag11 > 0
    cone.schur = schur - t1 / u
    @assert cone.schur > 0

    if cone.is_complex
        cvec_to_rvec!(cone.edge, edge)
        cvec_to_rvec!(cone.invedge, invedge)
    end

    cone.hess_inv_hess_updated = true
    return
end

# symmetric arrow matrix
function update_hess(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    if !cone.hess_inv_hess_updated
        update_hess_inv_hess(cone)
    end
    n = cone.n

    if !isdefined(cone, :hess)
        # initialize sparse idxs for upper triangle of Hessian
        dim = cone.dim
        H_nnz_tri = 2 * dim - 1 + (cone.is_complex ? n : 0)
        I = Vector{Int}(undef, H_nnz_tri)
        J = Vector{Int}(undef, H_nnz_tri)
        idxs1 = 1:dim
        I[idxs1] .= 1
        J[idxs1] .= idxs1
        idxs2 = (dim + 1):(2 * dim - 1)
        I[idxs2] .= 2:dim
        J[idxs2] .= 2:dim
        if cone.is_complex
            idxs3 = (2 * dim):H_nnz_tri
            I[idxs3] .= 2:2:dim
            J[idxs3] .= 3:2:dim
        end
        V = ones(T, H_nnz_tri)
        cone.hess = Symmetric(sparse(I, J, V, dim, dim), :U)
    end

    # modify nonzeros of sparse data structure of upper triangle of Hessian
    H_nzval = cone.hess.data.nzval
    H_nzval[1] = cone.diag11
    nz_idx = 2
    diag_idx = 1
    @inbounds for j in 1:n
        H_nzval[nz_idx] = cone.edge[diag_idx]
        H_nzval[nz_idx + 1] = cone.diag[diag_idx]
        nz_idx += 2
        diag_idx += 1
        if cone.is_complex
            H_nzval[nz_idx] = cone.edge[diag_idx]
            H_nzval[nz_idx + 1] = cone.offdiag[j]
            H_nzval[nz_idx + 2] = cone.diag[diag_idx]
            nz_idx += 3
            diag_idx += 1
        end
    end

    cone.hess_updated = true
    return cone.hess
end

# Diag(0, inv(diag)) + xx' / schur, where x = (-1, edge ./ diag)
function update_inv_hess(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    if !cone.hess_inv_hess_updated
        update_hess_inv_hess(cone)
    end
    Hi = cone.inv_hess.data

    Hi[1, 1] = 1
    Hi[1, 2:end] .= cone.invedge
    @inbounds for j in 2:cone.dim, i in 2:j
        Hi[i, j] = Hi[1, j] * Hi[1, i]
    end
    Hi ./= cone.schur
    if cone.is_complex
        @inbounds for (j, oj) in enumerate(cone.offdiag)
            detj = cone.detdiag[j]
            Hi[2j, 2j] += cone.diag[2j] / detj
            Hi[2j + 1, 2j + 1] += cone.diag[2j - 1] / detj
            Hi[2j, 2j + 1] -= oj / detj
        end
    else
        @inbounds for (j, dj) in enumerate(cone.diag)
            Hi[j + 1, j + 1] += inv(dj)
        end
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

# uses edge, diag11, diag, offdiag
function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    if !cone.hess_inv_hess_updated
        update_hess_inv_hess(cone)
    end

    @views copyto!(prod[1, :], arr[1, :])
    @views mul!(prod[1, :], arr[2:end, :]', cone.edge, true, cone.diag11)
    @views mul!(prod[2:end, :], cone.edge, arr[1, :]')
    @. @views prod[2:end, :] += cone.diag * arr[2:end, :]
    if cone.is_complex
        @inbounds for (j, oj) in enumerate(cone.offdiag)
            @. @views prod[2j, :] += oj * arr[2j + 1, :]
            @. @views prod[2j + 1, :] += oj * arr[2j, :]
        end
    end

    return prod
end

# uses invedge, schur, diag, offdiag, det
function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    if !cone.hess_inv_hess_updated
        update_hess_inv_hess(cone)
    end

    @views copyto!(prod[1, :], arr[1, :])
    @views mul!(prod[1, :], arr[2:end, :]', cone.invedge, true, true)
    @. @views prod[2:end, :] = cone.invedge * prod[1, :]'
    prod ./= cone.schur
    if cone.is_complex
        @inbounds for (j, oj) in enumerate(cone.offdiag)
            detj = cone.detdiag[j]
            d1j = cone.diag[2j - 1]
            d2j = cone.diag[2j]
            @. @views prod[2j, :] += (d2j * arr[2j, :] - oj * arr[2j + 1, :]) / detj
            @. @views prod[2j + 1, :] += (d1j * arr[2j + 1, :] - oj * arr[2j, :]) / detj
        end
    else
        @. @views prod[2:end, :] += arr[2:end, :] / cone.diag
    end

    return prod
end

# multiply by a sparse sqrt of hessian
function hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    if !cone.hess_inv_hess_updated
        update_hess_inv_hess(cone) # TODO needed?
    end
    @. cone.rtdiag = sqrt(cone.diag) # TODO update

    @. @views prod[1, :] = sqrt(cone.schur) * arr[1, :]
    if cone.is_complex
        for (j, oj) in enumerate(cone.offdiag)
            # TODO cache these fields?
            erj = cone.edge[2j - 1]
            eij = cone.edge[2j]
            rtd1j = sqrt(cone.diag[2j - 1])
            rtdetj = sqrt(cone.detdiag[j])
            ortd1j = oj / rtd1j
            side1j = erj / rtd1j
            side2j = (eij * rtd1j - erj * ortd1j) / rtdetj
            rtdetd1j = rtdetj / rtd1j
            @. @views prod[2j, :] = side1j * arr[1, :] + rtd1j * arr[2j, :] + ortd1j * arr[2j + 1, :]
            @. @views prod[2j + 1, :] = side2j * arr[1, :] + rtdetd1j * arr[2j + 1, :]
        end
    else
        @. @views prod[2:end, :] = cone.edge / cone.rtdiag * arr[1, :]' + cone.rtdiag * arr[2:end, :]
    end

    return prod
end

# multiply by sparse U factor of inverse hessian
function inv_hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    if !cone.hess_inv_hess_updated
        update_hess_inv_hess(cone) # TODO needed?
    end
    @. cone.rtdiag = sqrt(cone.diag) # TODO update

    @. @views prod[1, :] = arr[1, :]
    @views mul!(prod[1, :], arr[2:end, :]', cone.invedge, true, true)
    prod[1, :] ./= sqrt(cone.schur)
    if cone.is_complex
        for (j, oj) in enumerate(cone.offdiag)
            # TODO cache these fields?
            rtd2j = sqrt(cone.diag[2j])
            rtdetj = sqrt(cone.detdiag[j])
            rtd2detj = rtd2j / rtdetj
            ortd2detj = oj / rtd2j / rtdetj
            @. @views prod[2j, :] = rtd2detj * arr[2j, :] - ortd2detj * arr[2j + 1, :]
            @. @views prod[2j + 1, :] = arr[2j + 1, :] / rtd2j
        end
    else
        @. @views prod[2:end, :] = arr[2:end, :] / cone.rtdiag
    end

    return prod
end

hess_nz_count(cone::EpiNormInf{<:Real, <:Real}) = 3 * cone.dim - 2
hess_nz_count(cone::EpiNormInf{<:Real, <:Complex}) = 3 * cone.dim - 2 + 2 * cone.n
hess_nz_count_tril(cone::EpiNormInf{<:Real, <:Real}) = 2 * cone.dim - 1
hess_nz_count_tril(cone::EpiNormInf{<:Real, <:Complex}) = 2 * cone.dim - 1 + cone.n
hess_nz_idxs_col(cone::EpiNormInf{<:Real, <:Real}, j::Int) = (j == 1 ? (1:cone.dim) : [1, j])
hess_nz_idxs_col(cone::EpiNormInf{<:Real, <:Complex}, j::Int) = (j == 1 ? (1:cone.dim) : (iseven(j) ? [1, j, j + 1] : [1, j - 1, j]))
hess_nz_idxs_col_tril(cone::EpiNormInf{<:Real, <:Real}, j::Int) = (j == 1 ? (1:cone.dim) : [j])
hess_nz_idxs_col_tril(cone::EpiNormInf{<:Real, <:Complex}, j::Int) = (j == 1 ? (1:cone.dim) : (iseven(j) ? [j, j + 1] : [j]))
