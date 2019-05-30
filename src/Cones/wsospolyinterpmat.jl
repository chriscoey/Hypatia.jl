#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial matrix cone parametrized by interpolation points ipwt

definition and dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
=#

mutable struct WSOSPolyInterpMat{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    R::Int
    U::Int
    ipwt::Vector{Matrix{T}}

    point::AbstractVector{T}
    g::Vector{T}
    H::Matrix{T}
    H2::Matrix{T}
    Hi::Matrix{T}
    F
    mat::Vector{Matrix{T}}
    matfact::Vector{CholeskyPivoted{T, Matrix{T}}}
    tmp1::Vector{Matrix{T}}
    tmp2::Matrix{T}
    tmp3::Matrix{T}
    blockmats::Vector{Vector{Vector{Matrix{T}}}}
    blockfacts::Vector{Vector{CholeskyPivoted{T, Matrix{T}}}}
    PlambdaP::Matrix{T}

    function WSOSPolyInterpMat{T}(R::Int, U::Int, ipwt::Vector{Matrix{T}}, is_dual::Bool) where {T <: HypReal}
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == U
        end
        cone = new{T}()
        cone.use_dual = !is_dual # using dual barrier
        dim = U * div(R * (R + 1), 2)
        cone.dim = dim
        cone.R = R
        cone.U = U
        cone.ipwt = ipwt
        return cone
    end
end

WSOSPolyInterpMat{T}(R::Int, U::Int, ipwt::Vector{Matrix{T}}) where {T <: HypReal} = WSOSPolyInterpMat{T}(R, U, ipwt, false)

function setup_data(cone::WSOSPolyInterpMat{T}) where {T <: HypReal}
    dim = cone.dim
    U = cone.U
    R = cone.R
    ipwt = cone.ipwt
    cone.g = Vector{T}(undef, dim)
    cone.H = Matrix{T}(undef, dim, dim)
    cone.H2 = similar(cone.H)
    cone.Hi = similar(cone.H)
    cone.mat = [similar(cone.H, size(ipwtj, 2) * R, size(ipwtj, 2) * R) for ipwtj in ipwt]
    cone.matfact = Vector{CholeskyPivoted{T, Matrix{T}}}(undef, length(ipwt))
    cone.tmp1 = [similar(cone.H, size(ipwtj, 2), U) for ipwtj in ipwt]
    cone.tmp2 = similar(cone.H, U, U)
    cone.tmp3 = similar(cone.tmp2)
    cone.blockmats = [Vector{Vector{Matrix{T}}}(undef, R) for ipwtj in ipwt]
    for i in eachindex(ipwt), j in 1:R # TODO actually store 1 fewer (no diagonal) and also make this less confusing
        cone.blockmats[i][j] = Vector{Matrix{T}}(undef, j)
        for k in 1:j # TODO actually need to only go up to j-1
            L = size(ipwt[i], 2)
            cone.blockmats[i][j][k] = Matrix{T}(undef, L, L)
        end
    end
    cone.blockfacts = [Vector{CholeskyPivoted{T, Matrix{T}}}(undef, R) for _ in eachindex(ipwt)]
    cone.PlambdaP = Matrix{T}(undef, R * U,  R * U)
    return
end

get_nu(cone::WSOSPolyInterpMat) = cone.R * sum(size(ipwtj, 2) for ipwtj in cone.ipwt)

function set_initial_point(arr::AbstractVector{T}, cone::WSOSPolyInterpMat{T}) where {T <: HypReal}
    # sum of diagonal matrices with interpolant polynomial repeating on the diagonal
    idx = 1
    for i in 1:cone.R, j in 1:i
        arr[idx:(idx + cone.U - 1)] .= (i == j) ? one(T) : zero(T)
        idx += cone.U
    end
    return arr
end

_blockrange(inner::Int, outer::Int) = (outer * (inner - 1) + 1):(outer * inner)

# NOTE this is experimental code
function check_in_cone(cone::WSOSPolyInterpMat)
    # check_in_cone_nowinv(cone)
    check_in_cone_master(cone)
end

# TODO all views can be allocated just once in the cone definition (delete _blockrange too)
function check_in_cone_nowinv(cone::WSOSPolyInterpMat{T}) where {T <: HypReal}
    # @timeit "build mat" begin
    for j in eachindex(cone.ipwt)
        ipwtj = cone.ipwt[j]
        tmp1j = cone.tmp1[j]
        L = size(ipwtj, 2)
        mat = cone.mat[j]

        uo = 1
        for p in 1:cone.R, q in 1:p
            point_pq = cone.point[uo:(uo + cone.U - 1)] # TODO prealloc
            if p != q
                @. point_pq *= rt2i
            end
            @. tmp1j = ipwtj' * point_pq'

            rinds = _blockrange(p, L)
            cinds = _blockrange(q, L)
            mul!(view(mat, rinds, cinds), tmp1j, ipwtj)

            uo += cone.U
        end

        if !(blockcholesky!(cone, L, j))
            return false
        end
    end
    # end

    # @timeit "grad hess" begin
    cone.g .= zero(T)
    cone.H .= zero(T)
    for j in eachindex(cone.ipwt)
        ipwtj = cone.ipwt[j]
        L = size(ipwtj, 2)

        # perform L \ kron(ipwt)
        ldivp = _block_trisolve(cone, L, j)
        # ldivp' * ldivp, only fills upper triangle
        _mulblocks!(cone, ldivp, L)
        PlambdaP = Symmetric(cone.PlambdaP, :U)

        uo = 0
        for p in 1:cone.R, q in 1:p
            uo += 1
            fact = (p == q) ? one(T) : rt2
            rinds = _blockrange(p, cone.U)
            cinds = _blockrange(q, cone.U)
            idxs = _blockrange(uo, cone.U)

            cone.g[idxs] -= diag(PlambdaP[rinds, cinds]) .* fact

            uo2 = 0
            for p2 in 1:cone.R, q2 in 1:p2
                uo2 += 1
                if uo2 < uo
                    continue
                end

                rinds2 = _blockrange(p2, cone.U)
                cinds2 = _blockrange(q2, cone.U)
                idxs2 = _blockrange(uo2, cone.U)

                fact = xor(p == q, p2 == q2) ? rt2i : one(T)
                @. cone.H[idxs, idxs2] += PlambdaP[rinds, rinds2] * PlambdaP[cinds, cinds2] * fact

                if (p != q) || (p2 != q2)
                    @. cone.H[idxs, idxs2] += PlambdaP[rinds, cinds2] * PlambdaP[cinds, rinds2] * fact
                end
            end
        end
    end
    # end

    return factorize_hess(cone)
end

# res stored lower triangle
function blockcholesky!(cone::WSOSPolyInterpMat{T}, L::Int, j::Int) where {T <: HypReal}
    R = cone.R
    res = cone.blockmats[j]
    tmp = zeros(L, L)
    facts = cone.blockfacts[j]
    for r in 1:R
        tmp .= zero(T)
        # blocks on the diagonal come from cholesky decomposition after back-substitution, result stored in cone.blockfacts
        for k in 1:(r - 1)
            BLAS.syrk!('U', 'N', one(T), res[r][k], one(T), tmp)
        end
        diag_block = cone.mat[j][_blockrange(r, L), _blockrange(r, L)] - tmp
        F = cholesky!(Symmetric(diag_block, :U), Val(true), check = false)
        if !(isposdef(F))
            return false
        end
        facts[r] = F

        # blocks off the diagonal come from back-substitution, contigous blocks stored in cone.blockmats
        for s in (r + 1):R
            for k in 1:(r - 1)
                # tmp += res[r][k] * res[s][k]'
                BLAS.gemm!('N', 'T', one(T), res[r][k], res[s][k], one(T), tmp)
            end
            rhs = cone.mat[j][_blockrange(s, L), _blockrange(r, L)] - tmp
            res[s][r] = (facts[r].L \ view(rhs, facts[r].p, :))'
        end
    end
    return true
end

function _block_trisolve(cone::WSOSPolyInterpMat{T}, blocknum::Int, L::Int, j::Int) where {T <: HypReal}
    Lmat = cone.blockmats[j]
    R = cone.R
    U = cone.U
    Fvec = cone.blockfacts[j]
    resvec = zeros(R * L, U)
    tmp = zeros(L, U)
    resvec[_blockrange(blocknum, L), :] = Fvec[blocknum].L \ view(cone.ipwt[j]', Fvec[blocknum].p, :)
    for r in (blocknum + 1):R
        tmp .= zero(T)
        for s in blocknum:(r - 1)
            # tmp -= Lmat[r][s] * resvec[_blockrange(s, L), :]
            resblock = resvec[_blockrange(s, L), :]
            BLAS.gemm!('N', 'N', -one(T), Lmat[r][s], resblock, one(T), tmp)
        end
        resvec[_blockrange(r, L), :] = Fvec[r].L \ view(tmp, Fvec[r].p, :)
    end
    return resvec
end

# one block-column at a time on the RHS
function _block_trisolve(cone::WSOSPolyInterpMat{T}, L::Int, j::Int) where {T <: HypReal}
    R = cone.R
    U = cone.U
    resmat = zeros(R * L, R * U)
    for r in 1:R
        resmat[:, _blockrange(r, U)] = _block_trisolve(cone, r, L, j)
    end
    return resmat
end

# multiply lower triangular block matrix transposed by itself
function _mulblocks!(cone::WSOSPolyInterpMat{T}, mat::Matrix{T}, L::Int) where {T <: HypReal}
    # cone.PlambdaP = mat' * mat
    R = cone.R
    U = cone.U
    for i in 1:R
        rinds = _blockrange(i, U)
        for j in i:R
            cinds = _blockrange(j, U)
            tmp .= zero(T)
            # since mat is block lower triangular rows only from max(i,j) start making a nonzero contribution to the product
            mulrange = ((j - 1) * L + 1):(L * R)
            mul!(view(cone.PlambdaP, rinds, cinds), mat[mulrange, _blockrange(i, U)]',  mat[mulrange, _blockrange(j, U)])
        end
    end
    return nothing
end

function check_in_cone_master(cone::WSOSPolyInterpMat{T}) where {T <: HypReal}
    # @timeit "build mat" begin
    for j in eachindex(cone.ipwt)
        ipwtj = cone.ipwt[j]
        tmp1j = cone.tmp1[j]
        L = size(ipwtj, 2)
        mat = cone.mat[j]

        uo = 1
        for p in 1:cone.R, q in 1:p
            point_pq = cone.point[uo:(uo + cone.U - 1)] # TODO prealloc
            if p != q
                @. point_pq *= rt2i
            end
            @. tmp1j = ipwtj' * point_pq'

            rinds = _blockrange(p, L)
            cinds = _blockrange(q, L)
            mul!(view(mat, rinds, cinds), tmp1j, ipwtj)

            uo += cone.U
        end

        cone.matfact[j] = cholesky!(Symmetric(mat, :L), Val(true), check = false)
        if !isposdef(cone.matfact[j])
            return false
        end
    end
    # end

    # @timeit "grad hess" begin
    cone.g .= zero(T)
    cone.H .= zero(T)
    for j in eachindex(cone.ipwt)
        # @timeit "W_inv" begin
        W_inv_j = inv(cone.matfact[j])
        # end

        ipwtj = cone.ipwt[j]
        tmp1j = cone.tmp1[j]
        tmp2 = cone.tmp2
        tmp3 = cone.tmp3

        L = size(ipwtj, 2)
        uo = 0
        for p in 1:cone.R, q in 1:p
            uo += 1
            fact = (p == q) ? one(T) : rt2
            rinds = _blockrange(p, L)
            cinds = _blockrange(q, L)
            idxs = _blockrange(uo, cone.U)

            for i in 1:cone.U
                cone.g[idxs[i]] -= ipwtj[i, :]' * view(W_inv_j, rinds, cinds) * ipwtj[i, :] * fact
            end

            uo2 = 0
            for p2 in 1:cone.R, q2 in 1:p2
                uo2 += 1
                if uo2 < uo
                    continue
                end

                rinds2 = _blockrange(p2, L)
                cinds2 = _blockrange(q2, L)
                idxs2 = _blockrange(uo2, cone.U)

                mul!(tmp1j, view(W_inv_j, rinds, rinds2), ipwtj')
                mul!(tmp2, ipwtj, tmp1j)
                mul!(tmp1j, view(W_inv_j, cinds, cinds2), ipwtj')
                mul!(tmp3, ipwtj, tmp1j)
                fact = xor(p == q, p2 == q2) ? rt2i : one(T)
                @. cone.H[idxs, idxs2] += tmp2 * tmp3 * fact

                if (p != q) || (p2 != q2)
                    mul!(tmp1j, view(W_inv_j, rinds, cinds2), ipwtj')
                    mul!(tmp2, ipwtj, tmp1j)
                    mul!(tmp1j, view(W_inv_j, cinds, rinds2), ipwtj')
                    mul!(tmp3, ipwtj, tmp1j)
                    @. cone.H[idxs, idxs2] += tmp2 * tmp3 * fact
                end
            end
        end
    end
    # end

    return factorize_hess(cone)
end
