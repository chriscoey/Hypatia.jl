#=
Copyright 2018, Chris Coey and contributors
=#

using Test
import Random
using LinearAlgebra
using SparseArrays
import GenericLinearAlgebra.svdvals!
import Hypatia
import Hypatia.HypReal
import Hypatia.Solvers.build_solve_check
const CO = Hypatia.Cones

function dimension1(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    c = T[-1, 0]
    A = zeros(T, 0, 2)
    b = T[]
    G = T[1 0]
    h = T[1]
    cones = CO.Cone{T}[CO.Nonnegative{T}(1, false)]
    cone_idxs = [1:1]

    for use_sparse in (false, true)
        if use_sparse
            if T != Float64
                continue # TODO currently cannot do preprocessing with sparse A or G if not using Float64
            end
            A = sparse(A)
            G = sparse(G)
        end
        r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -1 atol=tol rtol=tol
        @test r.x ≈ [1, 0] atol=tol rtol=tol
        @test isempty(r.y)

        @test_throws ErrorException test_options.linear_model{T}(T[-1, -1], A, b, G, h, cones, cone_idxs)
    end
end

function consistent1(T, test_options)
    Random.seed!(1)
    (n, p, q) = (30, 15, 30)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    G = Matrix{T}(I, q, n)
    rnd1 = rand(T)
    rnd2 = rand(T)
    A[11:15, :] = rnd1 * A[1:5, :] - rnd2 * A[6:10, :]
    b = vec(sum(A, dims = 2))
    rnd1 = rand(T)
    rnd2 = rand(T)
    A[:, 11:15] = rnd1 * A[:, 1:5] - rnd2 * A[:, 6:10]
    G[:, 11:15] = rnd1 * G[:, 1:5] - rnd2 * G[:, 6:10]
    c[11:15] = rnd1 * c[1:5] - rnd2 * c[6:10]
    h = zeros(T, q)
    cones = CO.Cone{T}[CO.Nonpositive{T}(q)]
    cone_idxs = [1:q]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
end

function inconsistent1(T, test_options)
    Random.seed!(1)
    (n, p, q) = (30, 15, 30)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    G = Matrix{T}(-I, q, n)
    b = rand(T, p)
    rnd1 = rand(T)
    rnd2 = rand(T)
    A[11:15, :] = rnd1 * A[1:5, :] - rnd2 * A[6:10, :]
    b[11:15] = 2 * (rnd1 * b[1:5] - rnd2 * b[6:10])
    h = zeros(T, q)

    @test_throws ErrorException test_options.linear_model{T}(c, A, b, G, h, CO.Cone{T}[CO.Nonnegative{T}(q)], [1:q])
end

function inconsistent2(T, test_options)
    Random.seed!(1)
    (n, p, q) = (30, 15, 30)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    G = Matrix{T}(-I, q, n)
    b = rand(T, p)
    rnd1 = rand(T)
    rnd2 = rand(T)
    A[:,11:15] = rnd1 * A[:,1:5] - rnd2 * A[:,6:10]
    G[:,11:15] = rnd1 * G[:,1:5] - rnd2 * G[:,6:10]
    c[11:15] = 2 * (rnd1 * c[1:5] - rnd2 * c[6:10])
    h = zeros(T, q)

    @test_throws ErrorException test_options.linear_model{T}(c, A, b, G, h, CO.Cone{T}[CO.Nonnegative{T}(q)], [1:q])
end

function orthant1(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Random.seed!(1)
    (n, p, q) = (6, 3, 6)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    b = vec(sum(A, dims = 2))
    h = zeros(T, q)
    cone_idxs = [1:q]

    # nonnegative cone
    G = SparseMatrixCSC(-one(T) * I, q, n)
    cones = CO.Cone{T}[CO.Nonnegative{T}(q)]
    rnn = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test rnn.status == :Optimal

    # nonpositive cone
    G = SparseMatrixCSC(one(T) * I, q, n)
    cones = CO.Cone{T}[CO.Nonpositive{T}(q)]
    rnp = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test rnp.status == :Optimal

    @test rnp.primal_obj ≈ rnn.primal_obj atol=tol rtol=tol
end

function orthant2(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    b = vec(sum(A, dims = 2))
    G = rand(T, q, n) - Matrix(T(2) * I, q, n)
    h = G * ones(T, n)
    cone_idxs = [1:q]

    # use dual barrier
    cones = CO.Cone{T}[CO.Nonnegative{T}(q, true)]
    r1 = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r1.status == :Optimal

    # use primal barrier
    cones = CO.Cone{T}[CO.Nonnegative{T}(q, false)]
    r2 = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r2.status == :Optimal

    @test r1.primal_obj ≈ r2.primal_obj atol=tol rtol=tol
end

function orthant3(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Random.seed!(1)
    (n, p, q) = (15, 6, 15)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    b = vec(sum(A, dims = 2))
    G = Diagonal(one(T) * I, n)
    h = zeros(T, q)
    cone_idxs = [1:q]

    # use dual barrier
    cones = CO.Cone{T}[CO.Nonpositive{T}(q, true)]
    r1 = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r1.status == :Optimal

    # use primal barrier
    cones = CO.Cone{T}[CO.Nonpositive{T}(q, false)]
    r2 = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r2.status == :Optimal

    @test r1.primal_obj ≈ r2.primal_obj atol=tol rtol=tol
end

function orthant4(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    b = vec(sum(A, dims = 2))
    G = rand(T, q, n) - Matrix(T(2) * I, q, n)
    h = vec(sum(G, dims = 2))

    # mixture of nonnegative and nonpositive cones
    cones = CO.Cone{T}[CO.Nonnegative{T}(4, false), CO.Nonnegative{T}(6, true)]
    cone_idxs = [1:4, 5:10]
    r1 = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r1.status == :Optimal

    # only nonnegative cone
    cones = CO.Cone{T}[CO.Nonnegative{T}(10, false)]
    cone_idxs = [1:10]
    r2 = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r2.status == :Optimal

    @test r1.primal_obj ≈ r2.primal_obj atol=tol rtol=tol
end

function epinorminf1(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Tirt2 = inv(sqrt(T(2)))
    c = T[0, -1, -1]
    A = T[1 0 0; 0 1 0]
    b = [one(T), Tirt2]
    G = SparseMatrixCSC(-one(T) * I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiNormInf{T}(3)]
    cone_idxs = [1:3]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1 - Tirt2 atol=tol rtol=tol
    @test r.x ≈ [1, Tirt2, 1] atol=tol rtol=tol
    @test r.y ≈ [1, 1] atol=tol rtol=tol
end

function epinorminf2(T, test_options)
    tol = max(1e-5, 10 * sqrt(sqrt(eps(T))))
    l = 3
    L = 2l + 1
    c = collect(T, -l:l)
    A = spzeros(T, 2, L)
    A[1, 1] = A[1, L] = A[2, 1] = 1; A[2, L] = -1
    b = T[0, 0]
    G = [spzeros(T, 1, L); sparse(one(T) * I, L, L); spzeros(T, 1, L); sparse(T(2) * I, L, L)]
    h = zeros(T, 2L + 2); h[1] = 1; h[L + 2] = 1
    cones = CO.Cone{T}[CO.EpiNormInf{T}(L + 1, true), CO.EpiNormInf{T}(L + 1, false)]
    cone_idxs = [1:(L + 1), (L + 2):(2L + 2)]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -l + 1 atol=tol rtol=tol
    @test r.x[2] ≈ 0.5 atol=tol rtol=tol
    @test r.x[end - 1] ≈ -0.5 atol=tol rtol=tol
    @test sum(abs, r.x) ≈ 1 atol=tol rtol=tol
end

function epinorminf3(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    c = T[1, 0, 0, 0, 0, 0]
    A = zeros(T, 0, 6)
    b = zeros(T, 0)
    G = Diagonal(-one(T) * I, 6)
    h = zeros(T, 6)
    cones = CO.Cone{T}[CO.EpiNormInf{T}(6)]
    cone_idxs = [1:6]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
end

function epinorminf4(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    c = T[0, 1, -1]
    A = T[1 0 0; 0 1 0]
    b = T[1, -0.4]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiNormInf{T}(3, true)]
    cone_idxs = [1:3]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1 atol=tol rtol=tol
    @test r.x ≈ [1, -0.4, 0.6] atol=tol rtol=tol
    @test r.y ≈ [1, 0] atol=tol rtol=tol
end

function epinorminf5(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Random.seed!(1)
    c = T[1, 0, 0, 0, 0, 0]
    A = rand(T(-9):T(9), 3, 6)
    b = vec(sum(A, dims = 2))
    G = rand(T, 6, 6)
    h = vec(sum(G, dims = 2))
    cones = CO.Cone{T}[CO.EpiNormInf{T}(6, true)]
    cone_idxs = [1:6]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 1 atol=tol rtol=tol
end

function epinormeucl1(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Trt2 = sqrt(T(2))
    Tirt2 = inv(Trt2)
    c = T[0, -1, -1]
    A = T[1 0 0; 0 1 0]
    b = T[1, Tirt2]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)
    cone_idxs = [1:3]

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiNormEucl{T}(3, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -Trt2 atol=tol rtol=tol
        @test r.x ≈ [1, Tirt2, Tirt2] atol=tol rtol=tol
        @test r.y ≈ [Trt2, 0] atol=tol rtol=tol
    end
end

function epinormeucl2(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    c = T[0, -1, -1]
    A = T[1 0 0]
    b = T[0]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cone_idxs = [1:3]

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiNormEucl{T}(3, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function epipersquare1(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    c = T[0, 0, -1, -1]
    A = T[1 0 0 0; 0 1 0 0]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 4, 4)
    h = zeros(T, 4)
    cone_idxs = [1:4]

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiPerSquare{T}(4, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -sqrt(T(2)) atol=tol rtol=tol
        @test r.x[3:4] ≈ [1, 1] / sqrt(T(2)) atol=tol rtol=tol
    end
end

function epipersquare2(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Tirt2 = inv(sqrt(T(2)))
    c = T[0, 0, -1]
    A = T[1 0 0; 0 1 0]
    b = T[Tirt2 / 2, Tirt2]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)
    cone_idxs = [1:3]

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiPerSquare{T}(3, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -Tirt2 atol=tol rtol=tol
        @test r.x[2] ≈ Tirt2 atol=tol rtol=tol
    end
end

function epipersquare3(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    c = T[0, 1, -1, -1]
    A = T[1 0 0 0]
    b = T[0]
    G = SparseMatrixCSC(-one(T) * I, 4, 4)
    h = zeros(T, 4)
    cone_idxs = [1:4]

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiPerSquare{T}(4, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function semidefinite1(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Trt2 = sqrt(T(2))
    Trt2i = inv(Trt2)
    c = T[0, -1, 0]
    A = T[1 0 0; 0 0 1]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)
    cone_idxs = [1:3]

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(3, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
        @test r.status == :Optimal
        if is_dual
            @test r.primal_obj ≈ -Trt2 atol=tol rtol=tol
            @test r.x[2] ≈ Trt2 atol=tol rtol=tol
        else
            @test r.primal_obj ≈ -Trt2i atol=tol rtol=tol
            @test r.x[2] ≈ Trt2i atol=tol rtol=tol
        end
    end
end

function semidefinite2(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    c = T[0, -1, 0]
    A = T[1 0 1]
    b = T[0]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cone_idxs = [1:3]

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(3, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function semidefinitecomplex1(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Trt2 = sqrt(T(2))
    c = T[1, 0, 0, 1]
    A = T[0 0 1 0]
    b = T[1]
    G = Diagonal(-one(T) * I, 4)
    h = zeros(T, 4)
    cone_idxs = [1:4]
    cones = CO.Cone{T}[CO.PosSemidefTri{T, Complex{T}}(4)]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 2 atol=tol rtol=tol
    @test r.x ≈ [1, 0, 1, 1] atol=tol rtol=tol
end

function hypoperlog1(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Texph = exp(T(0.5))
    c = T[1, 1, 1]
    A = T[0 1 0; 1 0 0]
    b = T[2, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.HypoPerLog{T}(3)]
    cone_idxs = [1:3]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 2 * Texph + 3 atol=tol rtol=tol
    @test r.x ≈ [1, 2, 2 * Texph] atol=tol rtol=tol
    @test r.y ≈ -[1 + Texph / 2, 1 + Texph] atol=tol rtol=tol
    @test r.z ≈ c + A' * r.y atol=tol rtol=tol
end

function hypoperlog2(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    c = T[-1, 0, 0]
    A = T[0 1 0]
    b = T[0]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.HypoPerLog{T}(3)]
    cone_idxs = [1:3]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
end

function hypoperlog3(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    c = T[1, 1, 1]
    A = zeros(T, 0, 3)
    b = zeros(T, 0)
    G = sparse([1, 2, 3, 4], [1, 2, 3, 1], -ones(T, 4))
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.HypoPerLog{T}(3), CO.Nonnegative{T}(1)]
    cone_idxs = [1:3, 4:4]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
    @test isempty(r.y)
end

function hypoperlog4(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Texp2 = exp(T(-2))
    c = T[0, 0, 1]
    A = T[0 1 0; 1 0 0]
    b = T[1, -1]
    G = SparseMatrixCSC(-one(T) * I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.HypoPerLog{T}(3, true)]
    cone_idxs = [1:3]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ Texp2 atol=tol rtol=tol
    @test r.x ≈ [-1, 1, Texp2] atol=tol rtol=tol
end

function hypoperlog5(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Tlogq = log(T(0.25))
    c = T[-1, 0, 0]
    A = T[0 1 1]
    b = T[1]
    G = sparse([1, 3, 4], [1, 2, 3], -ones(T, 3))
    h = T[0, 1, 0, 0]
    cones = CO.Cone{T}[CO.HypoPerLog{T}(4)]
    cone_idxs = [1:4]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -Tlogq atol=tol rtol=tol
    @test r.x ≈ [Tlogq, 0.5, 0.5] atol=tol rtol=tol
    @test r.y ≈ [2] atol=tol rtol=tol
end

function hypoperlog6(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    c = T[-1, 0, 0]
    A = zeros(T, 0, 3)
    b = zeros(T, 0)
    G = sparse([1, 3, 4], [1, 2, 3], -ones(T, 3))
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.HypoPerLog{T}(4)]
    cone_idxs = [1:4]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test r.x[1] ≈ 0 atol=tol rtol=tol
    @test isempty(r.y)
end

function epiperpower1(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    c = T[0, 0, -1]
    A = T[1 0 0; 0 1 0]
    b = T[0.5, 1]
    G = Diagonal(-T(10) * I, 3)
    h = zeros(T, 3)
    cone_idxs = [1:3]

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiPerPower{T}(T(2), is_dual)]

        r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? -sqrt(T(2)) : -inv(sqrt(T(2)))) atol=tol rtol=tol
        @test r.x[1:2] ≈ [0.5, 1] atol=tol rtol=tol
    end
end

function epiperpower2(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    c = T[0, 0, 1]
    A = T[1 0 0; 0 1 0]
    b = T[0, 1]
    G = SparseMatrixCSC(-T(100) * I, 3, 3)
    h = zeros(T, 3)
    cone_idxs = [1:3]

    for is_dual in (true, false), alpha in T[1.5, 2.5]
        cones = CO.Cone{T}[CO.EpiPerPower{T}(alpha, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test r.x[1:2] ≈ [0, 1] atol=tol rtol=tol
    end
end

function hypogeomean1(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    c = T[-1, 0, 0]
    A = T[0 0 1; 0 1 0]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)
    cone_idxs = [1:3]

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.HypoGeomean{T}(ones(T, 2) / 2, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? 0 : -inv(sqrt(T(2)))) atol=tol rtol=tol
        @test r.x[2:3] ≈ [1, 0.5] atol=tol rtol=tol
    end
end

function hypogeomean2(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    l = 4
    c = vcat(zero(T), ones(T, l))
    A = T[one(T) zeros(T, 1, l)]
    G = SparseMatrixCSC(-one(T) * I, l + 1, l + 1)
    h = zeros(T, l + 1)
    cone_idxs = [1:(l + 1)]

    for is_dual in (true, false)
        b = is_dual ? [-one(T)] : [one(T)]
        cones = CO.Cone{T}[CO.HypoGeomean{T}(fill(inv(T(l)), l), is_dual)]

        r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? 1 : l) atol=tol rtol=tol
        @test r.x[2:end] ≈ (is_dual ? fill(inv(T(l)), l) : ones(l)) atol=tol rtol=tol
    end
end

function hypogeomean3(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    l = 4
    c = ones(T, l)
    A = zeros(T, 0, l)
    b = zeros(T, 0)
    G = [zeros(T, 1, l); Matrix{T}(-I, l, l)]
    h = zeros(T, l + 1)
    cone_idxs = [1:(l + 1)]

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.HypoGeomean{T}(fill(inv(T(l)), l), is_dual)]

        r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function epinormspectral1(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Random.seed!(1)
    (Xn, Xm) = (3, 4)
    Xnm = Xn * Xm
    c = vcat(one(T), zeros(T, Xnm))
    A = hcat(zeros(T, Xnm, 1), Matrix{T}(I, Xnm, Xnm))
    b = rand(T, Xnm)
    G = Matrix{T}(-I, Xnm + 1, Xnm + 1)
    h = vcat(zero(T), rand(T, Xnm))
    cone_idxs = [1:(Xnm + 1)]

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiNormSpectral{T}(Xn, Xm, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
        @test r.status == :Optimal
        if is_dual
            @test sum(svdvals!(reshape(r.s[2:end], Xn, Xm))) ≈ r.s[1] atol=tol rtol=tol
            @test svdvals!(reshape(r.z[2:end], Xn, Xm))[1] ≈ r.z[1] atol=tol rtol=tol
        else
            @test svdvals!(reshape(r.s[2:end], Xn, Xm))[1] ≈ r.s[1] atol=tol rtol=tol
            @test sum(svdvals!(reshape(r.z[2:end], Xn, Xm))) ≈ r.z[1] atol=tol rtol=tol
        end
    end
end

function hypoperlogdet1(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Random.seed!(1)
    side = 4
    dim = 2 + div(side * (side + 1), 2)
    c = T[-1, 0]
    A = T[0 1]
    b = T[1]
    G = Matrix{T}(-I, dim, 2)
    mat_half = rand(T, side, side)
    mat = mat_half * mat_half'
    h = zeros(T, dim)
    CO.mat_U_to_vec!(view(h, 3:dim), mat)
    cones = CO.Cone{T}[CO.HypoPerLogdetTri{T}(dim)]
    cone_idxs = [1:dim]
    unscale = [(i == j ? one(T) : inv(T(2))) for i in 1:side for j in 1:i]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.x[1] ≈ -r.primal_obj atol=tol rtol=tol
    @test r.x[2] ≈ 1 atol=tol rtol=tol
    sol_mat = zeros(T, side, side)
    CO.vec_to_mat_U!(sol_mat, r.s[3:end])
    @test r.s[2] * logdet(Symmetric(sol_mat, :U) / r.s[2]) ≈ r.s[1] atol=tol rtol=tol
    CO.vec_to_mat_U!(sol_mat, -r.z[3:end] .* unscale)
    @test r.z[1] * (logdet(Symmetric(sol_mat, :U) / r.z[1]) + T(side)) ≈ r.z[2] atol=tol rtol=tol
end

function hypoperlogdet2(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Random.seed!(1)
    side = 3
    dim = 2 + div(side * (side + 1), 2)
    c = T[0, 1]
    A = T[1 0]
    b = T[-1]
    G = Matrix{T}(-I, dim, 2)
    mat_half = rand(T, side, side)
    mat = mat_half * mat_half'
    h = zeros(T, dim)
    CO.mat_U_to_vec!(view(h, 3:dim), mat)
    cones = CO.Cone{T}[CO.HypoPerLogdetTri{T}(dim, true)]
    cone_idxs = [1:dim]
    unscale = [(i == j ? one(T) : inv(T(2))) for i in 1:side for j in 1:i]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.x[2] ≈ r.primal_obj atol=tol rtol=tol
    @test r.x[1] ≈ -1 atol=tol rtol=tol
    sol_mat = zeros(T, side, side)
    CO.vec_to_mat_U!(sol_mat, -r.s[3:end] .* unscale)
    @test r.s[1] * (logdet(Symmetric(sol_mat, :U) / r.s[1]) + T(side)) ≈ r.s[2] atol=tol rtol=tol
    CO.vec_to_mat_U!(sol_mat, r.z[3:end])
    @test r.z[2] * logdet(Symmetric(sol_mat, :U) / r.z[2]) ≈ r.z[1] atol=tol rtol=tol
end

function hypoperlogdet3(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    Random.seed!(1)
    side = 3
    dim = 2 + div(side * (side + 1), 2)
    c = T[-1, 0]
    A = T[0 1]
    b = T[0]
    G = SparseMatrixCSC(-one(T) * I, dim, 2)
    mat_half = rand(T, side, side)
    mat = mat_half * mat_half'
    h = zeros(T, dim)
    CO.mat_U_to_vec!(view(h, 3:dim), mat)
    cones = CO.Cone{T}[CO.HypoPerLogdetTri{T}(dim)]
    cone_idxs = [1:dim]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.x[1] ≈ -r.primal_obj atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
end

function epiperexp1(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    l = 5
    c = vcat(zero(T), -ones(T, l))
    A = hcat(one(T), zeros(T, 1, l))
    b = T[1]
    G = [-one(T) zeros(T, 1, l); zeros(T, 1, l + 1); zeros(T, l, 1) sparse(-one(T) * I, l, l)]
    h = zeros(T, l + 2)
    cones = CO.Cone{T}[CO.EpiPerExp{T}(l + 2)]
    cone_idxs = [1:(l + 2)]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.x[1] ≈ 1 atol=tol rtol=tol
    @test r.s[2] ≈ 0 atol=tol rtol=tol
    @test r.s[1] ≈ 1 atol=tol rtol=tol
end

function epiperexp2(T, test_options)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    l = 5
    c = vcat(zero(T), -ones(T, l))
    A = hcat(one(T), zeros(T, 1, l))
    b = T[1]
    G = [-one(T) spzeros(T, 1, l); spzeros(T, 1, l + 1); spzeros(T, l, 1) sparse(-one(T) * I, l, l)]
    h = zeros(T, l + 2); h[2] = 1
    cones = CO.Cone{T}[CO.EpiPerExp{T}(l + 2)]
    cone_idxs = [1:(l + 2)]

    r = build_solve_check(c, A, b, G, h, cones, cone_idxs; test_options...)
    @test r.status == :Optimal
    @test r.x[1] ≈ 1 atol=tol rtol=tol
    @test r.s[2] ≈ 1 atol=tol rtol=tol
    @test r.s[2] * sum(exp, r.s[3:end] / r.s[2]) ≈ r.s[1] atol=tol rtol=tol
end
