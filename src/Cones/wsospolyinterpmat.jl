#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial matrix cone parametrized by interpolation points ipwt

definition and dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
=#

mutable struct WSOSPolyInterpMat <: Cone
    use_dual::Bool
    dim::Int
    R::Int
    U::Int
    ipwt::Vector{Matrix{Float64}}
    point::Vector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    Hi::Matrix{Float64}
    F
    mat::Vector{Matrix{Float64}}
    matfact::Vector{CholeskyPivoted{Float64, Matrix{Float64}}}
    tmp1::Vector{Matrix{Float64}}
    tmp2::Matrix{Float64}
    tmp3::Matrix{Float64}

    function WSOSPolyInterpMat(R::Int, U::Int, ipwt::Vector{Matrix{Float64}}, is_dual::Bool)
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == U
        end
        cone = new()
        cone.use_dual = !is_dual # using dual barrier
        dim = U * div(R * (R + 1), 2)
        cone.dim = dim
        cone.R = R
        cone.U = U
        cone.ipwt = ipwt
        cone.point = similar(ipwt[1], dim)
        cone.g = similar(ipwt[1], dim)
        cone.H = similar(ipwt[1], dim, dim)
        cone.H2 = similar(cone.H)
        cone.Hi = similar(cone.H)
        cone.mat = [similar(ipwt[1], size(ipwtj, 2) * R, size(ipwtj, 2) * R) for ipwtj in ipwt]
        cone.matfact = Vector{CholeskyPivoted{Float64, Matrix{Float64}}}(undef, length(ipwt))
        cone.tmp1 = [similar(ipwt[1], size(ipwtj, 2), U) for ipwtj in ipwt]
        cone.tmp2 = similar(ipwt[1], U, U)
        cone.tmp3 = similar(cone.tmp2)
        return cone
    end
end

WSOSPolyInterpMat(R::Int, U::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterpMat(R, U, ipwt, false)

get_nu(cone::WSOSPolyInterpMat) = cone.R * sum(size(ipwtj, 2) for ipwtj in cone.ipwt)

function set_initial_point(arr::AbstractVector{Float64}, cone::WSOSPolyInterpMat)
    # sum of diagonal matrices with interpolant polynomial repeating on the diagonal
    idx = 1
    for i in 1:cone.R, j in 1:i
        arr[idx:(idx + cone.U - 1)] .= (i == j) ? 1.0 : 0.0
        idx += cone.U
    end
    return arr
end

_blockrange(inner::Int, outer::Int) = (outer * (inner - 1) + 1):(outer * inner)

# TODO all views can be allocated just once in the cone definition (delete _blockrange too)
function check_in_cone(cone::WSOSPolyInterpMat)
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
    cone.g .= 0.0
    cone.H .= 0.0
    for j in eachindex(cone.ipwt)
        # @timeit "W_inv" begin
        W_inv_j = inv(cone.matfact[j])
        # end

        ipwtj = cone.ipwt[j]
        tmp1j = cone.tmp1[j]
        tmp2 = cone.tmp2
        tmp3 = cone.tmp3

        L = size(ipwtj, 2)

        kron_ipwtj = kron(Matrix(I, cone.R, cone.R), ipwtj')
        kron_Winv = cone.matfact[j] \ kron_ipwtj
        # big_PLambdaP = kron(Matrix(I, cone.R, cone.R), ipwtj) * cone.matfact[j] \ kron(Matrix(I, cone.R, cone.R), ipwtj')
        # big_PLambdaP = kron(Matrix(I, cone.R, cone.R), ipwtj) * _block_uppertrisolve(cone.matfact[j].U, _block_lowertrisolve(cone.matfact[j].L, ipwtj, cone.R, L, cone.U), cone.R, L, cone.U)
        big_PLambdaP = PLmabdaP(cone.matfact[j], ipwtj, cone.R, cone.L, cone.U)

        uo = 0
        for p in 1:cone.R, q in 1:p
            uo += 1
            fact = (p == q) ? 1.0 : rt2
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
                @show tmp2 ./ big_PLambdaP[_blockrange(p, cone.U), _blockrange(p2, cone.U)]
                mul!(tmp1j, view(W_inv_j, cinds, cinds2), ipwtj')
                mul!(tmp3, ipwtj, tmp1j)
                fact = xor(p == q, p2 == q2) ? rt2i : 1.0
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

    # @timeit "inv hess" begin
    @. cone.H2 = cone.H
    cone.F = cholesky!(Symmetric(cone.H2, :U), Val(true), check = false)
    if !isposdef(cone.F)
        return false
    end
    cone.Hi .= inv(cone.F)
    # end

    return true
end


# left hand side always a kronecker with the identity
function _block_lowertrisolve(Lmat, ipwt, blocknum, R, L, U)
    resvec = zeros(R * L, U)
    resi(i) = resvec[_blockrange(i, L), :]
    Lmatij(i, j) = Lmat[_blockrange(i, L), _blockrange(j, L)]
    tmp = zeros(L, U)
    for r in 1:R
        if r == blocknum
            @show size(LowerTriangular(Lmatij(r, r)) \ ipwt'), size(resvec[_blockrange(r, L), :])
            resvec[_blockrange(r, L), :] = LowerTriangular(Lmatij(r, r)) \ ipwt'
        elseif r > blocknum
            tmp .= 0.0
            for s in blocknum:(r - 1)
                tmp -= Lmatij(r, s) * resi(s)
            end
            resvec[_blockrange(r, L), :] = LowerTriangular(Lmatij(r, r)) \ tmp
        end
    end
    return resvec
end
function _block_lowertrisolve(Lmat, ipwt, R, L, U)
    resmat = zeros(R * L, R * U)
    for r in 1:R
        resmat[:, _blockrange(r, U)] = _block_lowertrisolve(Lmat, ipwt, r, R, L, U)
    end
    return resmat
end
# left hand side always upper triangular block matrix
function _block_uppertrisolve(Umat, rhs, blocknum, R, L, U)
    resvec = zeros(R * L, U)
    resi(i) = resvec[_blockrange(i, L), :]
    Umatij(i, j) = Umat[_blockrange(i, L), _blockrange(j, L)]
    rhsi(i) = rhs[_blockrange(i, L), :]
    tmp = zeros(L, U)
    for r in reverse(1:R)
        if r == R
            resvec[_blockrange(r, L), :] = UpperTriangular(Umatij(r, r)) \ rhsi(r)
        else
            tmp .= rhsi(r)
            for s in reverse((r + 1):R)
                tmp -= Umatij(r, s) * resi(s)
            end
            resvec[_blockrange(r, L), :] = UpperTriangular(Umatij(r, r)) \ tmp
        end
    end
    return resvec
end
function _block_uppertrisolve(Umat, rhs, R, L, U)
    resmat = zeros(R * L, R * U)
    for r in 1:R
        resmat[:, _blockrange(r, U)] = _block_uppertrisolve(Umat, rhs[:, _blockrange(r, U)], r, R, L, U)
    end
    return resmat
end
function mul_ipwtkron(ipwt, x, R)
    res = Matrix(undef, R * U, R * U)
    for r in 1:R
        for s in 1:R # will actually be symmetric
            res[_blockrange(r, U), _blockrange(s, U)] = ipwt * x[_blockrange(r, L), _blockrange(s, U)]
        end
    end
    return res
end
function PLmabdaP(fact, ipwtj, R, L, U)
    mul_ipwtkron(ipwt, _block_uppertrisolve(fact.U, _block_lowertrisolve(fact.L, ipwtj, R, L, U), R, L, U), R)
end


# using Test
# using LinearAlgebra
# R = 3; U = 5; L = 4;
# ipwt = rand(U, L)
# kron_ipwt = kron(Matrix(I, R, R), ipwt)
# blocklambda = rand(R * L, R * L)
# blocklambda = blocklambda * blocklambda'
# F = cholesky(blocklambda)
# Ux = _block_lowertrisolve(F.L, ipwt, R, L, U)
# @test Ux ≈ F.L \ kron_ipwt'
# x = _block_uppertrisolve(F.U, Ux, R, L, U)
# @test x ≈ F \ kron_ipwt'
# @test kron_ipwt * inv(F) * kron_ipwt' ≈ mul_ipwtkron(ipwt, x, R)


inv_hess_prod!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::WSOSPolyInterpMat) = mul!(prod, Symmetric(cone.Hi, :U), arr)
