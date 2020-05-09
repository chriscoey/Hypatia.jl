#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

tests for primitive cone barrier oracles
=#

using Test
import Random
using LinearAlgebra
using SparseArrays
import ForwardDiff
import TimerOutputs
import Hypatia
const CO = Hypatia.Cones
const MU = Hypatia.ModelUtilities

timer = TimerOutputs.TimerOutput()

function test_barrier_oracles(
    cone::CO.Cone{T},
    barrier::Function;
    noise::T = T(0.1),
    scale::T = T(1e-2),
    tol::Real = 1000eps(T),
    init_tol::Real = tol,
    init_only::Bool = false,
    ) where {T <: Real}
    Random.seed!(1)

    CO.setup_data(cone)
    CO.set_timer(cone, timer)
    dim = CO.dimension(cone)
    point = Vector{T}(undef, dim)
    dual_point = copy(point)
    CO.set_initial_point(point, cone)
    CO.set_initial_point(dual_point, cone)
    @test load_reset_check(cone, point, dual_point)
    @test cone.point == point
    @test cone.dual_point == dual_point

    if isfinite(init_tol)
        # tests for centrality of initial point
        minus_grad = -CO.grad(cone)
        @test dot(point, minus_grad) ≈ norm(point) * norm(minus_grad) atol=init_tol rtol=init_tol
        @test point ≈ minus_grad atol=init_tol rtol=init_tol
        # @test CO.in_neighborhood(cone, minus_grad, one(T))
        @test CO.in_neighborhood(cone, zero(T))
    end
    init_only && return

    # perturb and scale the initial point and check feasible
    perturb_scale(point, dual_point, noise, scale)
    @test load_reset_check(cone, point, dual_point)

    # test gradient and Hessian oracles
    test_grad_hess(cone, point, dual_point, tol = tol)

    # check gradient and Hessian agree with ForwardDiff
    if dim < 10 # too slow if dimension is large
        grad = CO.grad(cone)
        fd_grad = ForwardDiff.gradient(barrier, point)
        @test grad ≈ fd_grad atol=tol rtol=tol

        hess = CO.hess(cone)
        fd_hess = ForwardDiff.hessian(barrier, point)
        @test hess ≈ fd_hess atol=tol rtol=tol
    end

    # TODO decide whether to add
    # # check 3rd order corrector agrees with ForwardDiff
    # # too slow if cone is too large or not using BlasReals
    # if CO.use_3order_corr(cone) && dim < 8 && T in (Float32, Float64)
    #     FD_3deriv = ForwardDiff.jacobian(x -> ForwardDiff.hessian(barrier, x), point)
    #     # check log-homog property that F'''(point)[point] = -2F''(point)
    #     @test reshape(FD_3deriv * point, dim, dim) ≈ -2 * hess
    #     # check correction term agrees with directional 3rd derivative
    #     primal_dir = perturb_scale(zeros(T, dim), noise, one(T))
    #     dual_dir = perturb_scale(zeros(T, dim), noise, one(T))
    #     Hinv_z = CO.inv_hess_prod!(similar(dual_dir), dual_dir, cone)
    #     FD_corr = reshape(FD_3deriv * primal_dir, dim, dim) * Hinv_z / -2
    #     @test FD_corr ≈ CO.correction(cone, primal_dir, dual_dir) atol=tol rtol=tol
    # end

    return
end

function test_grad_hess(cone::CO.Cone{T}, point::Vector{T}, dual_point::Vector{T}; tol::Real = 1000eps(T)) where {T <: Real}
    nu = CO.get_nu(cone)
    dim = length(point)
    grad = CO.grad(cone)
    hess = Matrix(CO.hess(cone))
    inv_hess = Matrix(CO.inv_hess(cone))

    @test dot(point, grad) ≈ -nu atol=tol rtol=tol
    @test hess * inv_hess ≈ I atol=tol rtol=tol

    prod_mat = similar(point, dim, dim)
    @test CO.hess_prod!(prod_mat, inv_hess, cone) ≈ I atol=tol rtol=tol
    @test CO.inv_hess_prod!(prod_mat, hess, cone) ≈ I atol=tol rtol=tol

    prod = similar(point)
    @test hess * point ≈ -grad atol=tol rtol=tol
    @test CO.hess_prod!(prod, point, cone) ≈ -grad atol=tol rtol=tol
    @test CO.inv_hess_prod!(prod, grad, cone) ≈ -point atol=tol rtol=tol

    prod_mat2 = Matrix(CO.hess_sqrt_prod!(prod_mat, inv_hess, cone)')
    @test CO.hess_sqrt_prod!(prod_mat, prod_mat2, cone) ≈ I atol=tol rtol=tol
    CO.inv_hess_sqrt_prod!(prod_mat2, Matrix(one(T) * I, dim, dim), cone)
    @test prod_mat2' * prod_mat2 ≈ inv_hess atol=tol rtol=tol

    if cone isa CO.HypoPerLog || cone isa CO.Nonnegative
        dual_grad = CO.dual_grad(cone)
        @test dot(dual_point, dual_grad) ≈ -nu atol=1000*tol rtol=1000*tol

        scal_hess = CO.scal_hess(cone, one(T))
        @test scal_hess * point ≈ dual_point
        @test scal_hess * dual_grad ≈ grad

        prod = similar(point)
        @test CO.scal_hess_prod!(prod, point, cone, one(T)) ≈ dual_point
        @test CO.scal_hess_prod!(prod, dual_grad, cone, one(T)) ≈ grad
    end

    mock_dual_point = -grad + T(1e-3) * randn(length(grad))
    CO.load_dual_point(cone, mock_dual_point)
    @test CO.in_neighborhood(cone, zero(T))

    return
end

function load_reset_check(cone::CO.Cone{T}, point::Vector{T}, dual_point::Vector{T}) where {T <: Real}
    CO.load_point(cone, point)
    CO.load_dual_point(cone, dual_point)
    CO.reset_data(cone)
    return CO.is_feas(cone)
end

function perturb_scale(point::Vector{T}, dual_point::Vector{T}, noise::T, scale::T) where {T <: Real}
    if !iszero(noise)
        @. point += 2 * noise * rand(T) - noise
        @. dual_point += 2 * noise * rand(T) - noise
    end
    if !isone(scale)
        point .*= scale
    end
    return point
end

# primitive cone barrier tests

function test_nonnegative_barrier(T::Type{<:Real})
    barrier = (s -> -sum(log, s))
    for dim in [1, 2, 3, 6]
        test_barrier_oracles(CO.Nonnegative{T}(dim), barrier)
    end
    return
end

function test_epinorminf_barrier(T::Type{<:Real})
    for n in [1, 2, 3, 5]
        # real epinorminf cone
        function R_barrier(s)
            (u, w) = (s[1], s[2:end])
            return -sum(log(abs2(u) - abs2(wj)) for wj in w) + (n - 1) * log(u)
        end
        test_barrier_oracles(CO.EpiNormInf{T, T}(1 + n), R_barrier)

        # complex epinorminf cone
        function C_barrier(s)
            (u, wr) = (s[1], s[2:end])
            w = CO.rvec_to_cvec!(zeros(Complex{eltype(s)}, n), wr)
            return -sum(log(abs2(u) - abs2(wj)) for wj in w) + (n - 1) * log(u)
        end
        test_barrier_oracles(CO.EpiNormInf{T, Complex{T}}(1 + 2n), C_barrier)
    end
    return
end

function test_epinormeucl_barrier(T::Type{<:Real})
    function barrier(s)
        (u, w) = (s[1], s[2:end])
        return -log(abs2(u) - sum(abs2, w))
    end
    for dim in [2, 3, 4, 6]
        test_barrier_oracles(CO.EpiNormEucl{T}(dim), barrier)
    end
    return
end

function test_epipersquare_barrier(T::Type{<:Real})
    function barrier(s)
        (u, v, w) = (s[1], s[2], s[3:end])
        return -log(2 * u * v - sum(abs2, w))
    end
    for dim in [3, 4, 5, 7]
        test_barrier_oracles(CO.EpiPerSquare{T}(dim), barrier)
    end
    return
end

function test_hypoperlog_barrier(T::Type{<:Real})
    function barrier(s)
        (u, v, w) = (s[1], s[2], s[3:end])
        return -log(v * sum(log(wj / v) for wj in w) - u) - sum(log, w) - length(w) * log(v)
    end
    for dim in [3, 5, 10]
        test_barrier_oracles(CO.HypoPerLog{T}(dim), barrier, init_tol = 1e-5)
    end
    for dim in [15, 65, 75, 100, 500]
        test_barrier_oracles(CO.HypoPerLog{T}(dim), barrier, init_tol = 1e-1, init_only = true)
    end
    return
end

function test_episumperentropy_barrier(T::Type{<:Real})
    for w_dim in [3, 4, 6]
        function barrier(s)
            (u, v, w) = (s[1], s[2:(w_dim + 1)], s[(w_dim + 2):dim])
            return -log(u - sum(wi * log(wi / vi) for (vi, wi) in zip(v, w))) - sum(log(vi) + log(wi) for (vi, wi) in zip(v, w))
        end
        dim = 1 + 2 * w_dim
        test_barrier_oracles(CO.EpiSumPerEntropy{T}(dim), barrier, init_tol = 1e-5)
    end
    for w_dim in [15, 65, 75, 100, 500]
        function barrier(s)
            (u, v, w) = (s[1], s[2:(w_dim + 1)], s[(w_dim + 2):dim])
            return -log(u - sum(wi * log(wi / vi) for (vi, wi) in zip(v, w))) - sum(log(vi) + log(wi) for (vi, wi) in zip(v, w))
        end
        dim = 1 + 2 * w_dim
        test_barrier_oracles(CO.EpiSumPerEntropy{T}(dim), barrier, init_tol = 1e-1, init_only = true)
    end
    return
end

function test_power_barrier(T::Type{<:Real})
    Random.seed!(1)
    for (m, n) in [(2, 1), (2, 3), (4, 1), (4, 4)]
        alpha = rand(T, m) .+ 1
        alpha ./= sum(alpha)
        function barrier(s)
            (u, w) = (s[1:m], s[(m + 1):end])
            return -log(prod(u[j] ^ (2 * alpha[j]) for j in eachindex(u)) - sum(abs2, w)) - sum((1 - alpha[j]) * log(u[j]) for j in eachindex(u))
        end
        test_barrier_oracles(CO.Power{T}(alpha, n), barrier)
    end
    return
end

function test_hypogeomean_barrier(T::Type{<:Real})
    Random.seed!(1)
    for dim in [2, 3, 5, 15, 90, 120, 500]
        alpha = rand(T, dim - 1) .+ 1
        alpha ./= sum(alpha)
        function barrier(s)
            (u, w) = (s[1], s[2:end])
            return -log(prod(w[j] ^ alpha[j] for j in eachindex(w)) - u) - sum(log(wi) for wi in w)
        end
        cone = CO.HypoGeomean{T}(alpha)
        if dim <= 3
            test_barrier_oracles(cone, barrier, init_tol = 1e-2)
        else
            test_barrier_oracles(cone, barrier, init_tol = 1e-2, init_only = true)
        end
        # test initial point when all alphas are the same
        cone = CO.HypoGeomean{T}(fill(inv(T(dim - 1)), dim - 1))
        test_barrier_oracles(cone, barrier, init_tol = sqrt(eps(T)), init_only = true)
    end
    return
end

function test_epinormspectral_barrier(T::Type{<:Real})
    for (n, m) in [(1, 1), (1, 2), (2, 2), (2, 4), (3, 4)]
        # real epinormspectral barrier
        function R_barrier(s)
            (u, W) = (s[1], reshape(s[2:end], n, m))
            return -logdet(cholesky!(Hermitian(abs2(u) * I - W * W'))) + (n - 1) * log(u)
        end
        test_barrier_oracles(CO.EpiNormSpectral{T, T}(n, m), R_barrier)

        # complex epinormspectral barrier
        function C_barrier(s)
            u = s[1]
            W = CO.rvec_to_cvec!(zeros(Complex{eltype(s)}, n, m), s[2:end])
            return -logdet(cholesky!(Hermitian(abs2(u) * I - W * W'))) + (n - 1) * log(u)
        end
        test_barrier_oracles(CO.EpiNormSpectral{T, Complex{T}}(n, m), C_barrier)
    end
    return
end

function test_matrixepipersquare_barrier(T::Type{<:Real})
    for (n, m) in [(1, 1), (1, 2), (2, 2), (2, 4), (3, 4)]
        # real matrixepipersquare barrier
        per_idx = CO.svec_length(n) + 1
        function R_barrier(s)
            U = CO.svec_to_smat!(similar(s, n, n), s[1:(per_idx - 1)], sqrt(T(2)))
            v = s[per_idx]
            W = reshape(s[(per_idx + 1):end], n, m)
            return -logdet(cholesky!(Symmetric(2 * v * U - W * W', :U))) + (n - 1) * log(v)
        end
        test_barrier_oracles(CO.MatrixEpiPerSquare{T, T}(n, m), R_barrier)

        # complex matrixepipersquare barrier
        per_idx = n ^ 2 + 1
        function C_barrier(s)
            U = CO.svec_to_smat!(zeros(Complex{eltype(s)}, n, n), s[1:(per_idx - 1)], sqrt(T(2)))
            v = s[per_idx]
            W = CO.rvec_to_cvec!(zeros(Complex{eltype(s)}, n, m), s[(per_idx + 1):end])
            return -logdet(cholesky!(Hermitian(2 * v * U - W * W', :U))) + (n - 1) * log(v)
        end
        test_barrier_oracles(CO.MatrixEpiPerSquare{T, Complex{T}}(n, m), C_barrier)
    end
    return
end

function test_linmatrixineq_barrier(T::Type{<:Real})
    Random.seed!(1)
    Rs_list = [[T, T], [Complex{T}, Complex{T}], [T, Complex{T}, T], [Complex{T}, T, T]]
    for side in [2, 3, 5], Rs in Rs_list
        As = Vector{LinearAlgebra.HermOrSym{R, Matrix{R}} where {R <: Hypatia.RealOrComplex{T}}}(undef, length(Rs))
        A_1_half = rand(Rs[1], side, side)
        As[1] = Hermitian(A_1_half * A_1_half' + I, :U)
        for i in 2:length(Rs)
            As[i] = Hermitian(rand(Rs[i], side, side), :U)
        end
        barrier(s) = -logdet(cholesky!(sum(s_i * A_i for (s_i, A_i) in zip(s, As))))
        test_barrier_oracles(CO.LinMatrixIneq{T}(As), barrier, init_tol = Inf)
    end
    return
end

function test_possemideftri_barrier(T::Type{<:Real})
    for side in [1, 2, 5]
        # real PSD cone
        function R_barrier(s)
            S = similar(s, side, side)
            CO.svec_to_smat!(S, s, sqrt(T(2)))
            return -logdet(cholesky!(Symmetric(S, :U)))
        end
        dim = CO.svec_length(side)
        test_barrier_oracles(CO.PosSemidefTri{T, T}(dim), R_barrier)

        # complex PSD cone
        function C_barrier(s)
            S = zeros(Complex{eltype(s)}, side, side)
            CO.svec_to_smat!(S, s, sqrt(T(2)))
            return -logdet(cholesky!(Hermitian(S, :U)))
        end
        dim = side^2
        test_barrier_oracles(CO.PosSemidefTri{T, Complex{T}}(dim), C_barrier)
    end
    return
end

function test_possemideftrisparse_barrier(T::Type{<:Real})
    if !(T <: LinearAlgebra.BlasReal)
        return # only works with BLAS real types
    end
    Random.seed!(1)
    invrt2 = inv(sqrt(T(2)))

    for side in [1, 2, 3, 5, 10, 20, 40, 80]
        # generate random sparsity pattern for lower triangle
        sparsity = inv(sqrt(side))
        (row_idxs, col_idxs, _) = findnz(tril!(sprand(Bool, side, side, sparsity)) + I)

        # real sparse PSD cone
        function R_barrier(s)
            scal_s = copy(s)
            for i in eachindex(s)
                if row_idxs[i] != col_idxs[i]
                    scal_s[i] *= invrt2
                end
            end
            S = Matrix(sparse(row_idxs, col_idxs, scal_s, side, side))
            return -logdet(cholesky(Symmetric(S, :L)))
        end
        test_barrier_oracles(CO.PosSemidefTriSparse{T, T}(side, row_idxs, col_idxs), R_barrier)

        # complex sparse PSD cone
        function C_barrier(s)
            scal_s = zeros(Complex{eltype(s)}, length(row_idxs))
            idx = 1
            for i in eachindex(scal_s)
                if row_idxs[i] == col_idxs[i]
                    scal_s[i] = s[idx]
                    idx += 1
                else
                    scal_s[i] = invrt2 * Complex(s[idx], s[idx + 1])
                    idx += 2
                end
            end
            S = Matrix(sparse(row_idxs, col_idxs, scal_s, side, side))
            return -logdet(cholesky!(Hermitian(S, :L)))
        end
        test_barrier_oracles(CO.PosSemidefTriSparse{T, Complex{T}}(side, row_idxs, col_idxs), C_barrier)
    end
    return
end

function test_hypoperlogdettri_barrier(T::Type{<:Real})
    for side in [1, 2, 3, 4, 5, 6, 12, 20]
        # real logdet barrier
        dim = 2 + CO.svec_length(side)
        cone = CO.HypoPerLogdetTri{T, T}(dim)
        function R_barrier(s)
            (u, v, W) = (s[1], s[2], zeros(eltype(s), side, side))
            CO.svec_to_smat!(W, s[3:end], sqrt(T(2)))
            return cone.sc_const * (-log(v * logdet(cholesky!(Symmetric(W / v, :U))) - u) - logdet(cholesky!(Symmetric(W, :U))) - (side + 1) * log(v))
        end
        if side <= 5
            test_barrier_oracles(cone, R_barrier, init_tol = 1e-5)
        else
            test_barrier_oracles(cone, R_barrier, init_tol = 1e-1, init_only = true)
        end

        # try sc_const = 1 (not self-concordant)
        cone = CO.HypoPerLogdetTri{T, T}(dim, sc_const = 1)
        function R_barrier_sc1(s)
            (u, v, W) = (s[1], s[2], zeros(eltype(s), side, side))
            CO.svec_to_smat!(W, s[3:end], sqrt(T(2)))
            return -log(v * logdet(cholesky!(Symmetric(W / v, :U))) - u) - logdet(cholesky!(Symmetric(W, :U))) - (side + 1) * log(v)
        end
        if side <= 3
            test_barrier_oracles(cone, R_barrier_sc1, init_tol = 1e-5)
        else
            test_barrier_oracles(cone, R_barrier_sc1, init_tol = 1e-1, init_only = true)
        end

        # complex logdet barrier
        dim = 2 + side^2
        cone = CO.HypoPerLogdetTri{T, Complex{T}}(dim)
        function C_barrier(s)
            (u, v, W) = (s[1], s[2], zeros(Complex{eltype(s)}, side, side))
            CO.svec_to_smat!(W, s[3:end], sqrt(T(2)))
            return cone.sc_const * (-log(v * logdet(cholesky!(Hermitian(W / v, :U))) - u) - logdet(cholesky!(Hermitian(W, :U))) - (side + 1) * log(v))
        end
        if side <= 4
            test_barrier_oracles(cone, C_barrier, init_tol = 1e-5)
        else
            test_barrier_oracles(cone, C_barrier, init_tol = 1e-1, init_only = true)
        end
    end
    return
end

function test_hyporootdettri_barrier(T::Type{<:Real})
    for side in [1, 2, 3, 5, 8]
        # real rootdet barrier
        dim = 1 + CO.svec_length(side)
        cone = CO.HypoRootdetTri{T, T}(dim)
        function R_barrier(s)
            (u, W) = (s[1], zeros(eltype(s), side, side))
            CO.svec_to_smat!(W, s[2:end], sqrt(T(2)))
            fact_W = cholesky!(Symmetric(W, :U))
            return cone.sc_const * (-log(exp(logdet(fact_W) / side) - u) - logdet(fact_W))
        end
        test_barrier_oracles(cone, R_barrier)

        # try sc_const = 1 (not self-concordant)
        dim = 1 + CO.svec_length(side)
        cone = CO.HypoRootdetTri{T, T}(dim, sc_const = 1)
        function R_barrier_sc1(s)
            (u, W) = (s[1], zeros(eltype(s), side, side))
            CO.svec_to_smat!(W, s[2:end], sqrt(T(2)))
            fact_W = cholesky!(Symmetric(W, :U))
            return -log(exp(logdet(fact_W) / side) - u) - logdet(fact_W)
        end
        test_barrier_oracles(cone, R_barrier_sc1)

        # complex rootdet barrier
        dim = 1 + side^2
        cone = CO.HypoRootdetTri{T, Complex{T}}(dim)
        function C_barrier(s)
            (u, W) = (s[1], zeros(Complex{eltype(s)}, side, side))
            CO.svec_to_smat!(W, s[2:end], sqrt(T(2)))
            fact_W = cholesky!(Hermitian(W, :U))
            return cone.sc_const * (-log(exp(logdet(fact_W) / side) - u) - logdet(fact_W))
        end
        test_barrier_oracles(cone, C_barrier)
    end
    return
end

function test_wsosinterpnonnegative_barrier(T::Type{<:Real})
    Random.seed!(1)
    for (n, halfdeg) in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1)]
        (U, _, Ps, _) = MU.interpolate(MU.Box{T}(-ones(T, n), ones(T, n)), halfdeg, sample = false) # use a unit box domain
        barrier(s) = -sum(logdet(cholesky!(Symmetric(P' * Diagonal(s) * P))) for P in Ps)
        cone = CO.WSOSInterpNonnegative{T, T}(U, Ps)
        test_barrier_oracles(cone, barrier, init_tol = Inf) # TODO center and test initial points
    end
    # TODO also test complex case CO.WSOSInterpNonnegative{T, Complex{T}} - need complex MU interp functions first
    return
end

function test_wsosinterppossemideftri_barrier(T::Type{<:Real})
    Random.seed!(1)
    rt2i = inv(sqrt(T(2)))
    for n in 1:3, halfdeg in 1:2, R in 1:3
        (U, _, Ps, _) = MU.interpolate(MU.Box{T}(-ones(T, n), ones(T, n)), halfdeg, sample = false)
        cone = CO.WSOSInterpPosSemidefTri{T}(R, U, Ps)
        function barrier(s)
            bar = zero(eltype(s))
            for P in Ps
                L = size(P, 2)
                Lambda = zeros(eltype(s), R * L, R * L)
                for i in 1:R, j in 1:i
                    Lambdaij = P' * Diagonal(s[CO.block_idxs(U, CO.svec_idx(i, j))]) * P
                    if i != j
                        Lambdaij .*= rt2i
                    end
                    Lambda[CO.block_idxs(L, i), CO.block_idxs(L, j)] = Lambdaij
                end
                bar -= logdet(cholesky!(Symmetric(Lambda, :L)))
            end
            return bar
        end
        test_barrier_oracles(cone, barrier, init_tol = Inf)
    end
    return
end

function test_wsosinterpepinormeucl_barrier(T::Type{<:Real})
    Random.seed!(1)
    for n in 1:3, halfdeg in 1:2, R in 2:3
        (U, _, Ps, _) = MU.interpolate(MU.Box{T}(-ones(T, n), ones(T, n)), halfdeg, sample = false)
        cone = CO.WSOSInterpEpiNormEucl{T}(R, U, Ps)
        function barrier(s)
            bar = zero(eltype(s))
            for P in Ps
                Lambda = P' * Diagonal(s[1:U]) * P
                Lambda1fact = cholesky!(Symmetric(copy(Lambda), :L))
                for i in 2:R
                    Lambdai = P' * Diagonal(s[CO.block_idxs(U, i)]) * P
                    ldiv!(Lambda1fact.L, Lambdai)
                    Lambda -= Lambdai' * Lambdai
                end
                bar -= logdet(cholesky!(Symmetric(Lambda))) + logdet(Lambda1fact)
            end
            return bar
        end
        test_barrier_oracles(cone, barrier, init_tol = Inf)
    end
    return
end
