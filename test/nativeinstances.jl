#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

native test instances

TODO
- improve efficiency of many of the tests by doing in-place linear algebra etc
- maybe pass in tol?
=#

using Test
import Random
using LinearAlgebra
import GenericLinearAlgebra.svdvals
import GenericLinearAlgebra.eigvals
using SparseArrays
import Hypatia
import Hypatia.Solvers.build_solve_check
const CO = Hypatia.Cones

function dimension1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0]
    A = zeros(T, 0, 2)
    b = T[]
    G = T[1 0]
    h = T[1]
    cones = CO.Cone{T}[CO.Nonnegative{T}(1)]

    for use_sparse in (false, true)
        if use_sparse
            A = sparse(A)
            G = sparse(G)
        end
        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -1 atol=tol rtol=tol
        @test r.x ≈ [1, 0] atol=tol rtol=tol
        @test isempty(r.y)

        @test_throws ErrorException options.linear_model{T}(T[-1, -1], A, b, G, h, cones)
    end
end

function consistent1(T; options...)
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
    cones = CO.Cone{T}[CO.Nonnegative{T}(q)]

    r = build_solve_check(c, A, b, G, h, cones; options...)
    @test r.status == :Optimal
end

function inconsistent1(T; options...)
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

    @test_throws ErrorException options.linear_model{T}(c, A, b, G, h, CO.Cone{T}[CO.Nonnegative{T}(q)])
end

function inconsistent2(T; options...)
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

    @test_throws ErrorException options.linear_model{T}(c, A, b, G, h, CO.Cone{T}[CO.Nonnegative{T}(q)])
end

function nonnegative1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (n, p, q) = (6, 3, 6)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    b = vec(sum(A, dims = 2))
    G = SparseMatrixCSC(-one(T) * I, q, n)
    h = zeros(T, q)
    cones = CO.Cone{T}[CO.Nonnegative{T}(q)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, obj_offset = one(T), options...)
    @test r.status == :Optimal
end

function nonnegative2(T; options...)
    tol = 2 * sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    b = vec(sum(A, dims = 2))
    G = rand(T, q, n) - Matrix(T(2) * I, q, n)
    h = vec(sum(G, dims = 2))
    cones = CO.Cone{T}[CO.Nonnegative{T}(q)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
end

function nonnegative3(T; options...)
    tol = 2 * sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (n, p, q) = (15, 6, 15)
    c = rand(T(0):T(9), n)
    A = rand(T(-9):T(9), p, n)
    b = vec(sum(A, dims = 2))
    G = Diagonal(-one(T) * I, n)
    h = zeros(T, q)
    cones = CO.Cone{T}[CO.Nonnegative{T}(q)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
end

function epinorminf1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Tirt2 = inv(sqrt(T(2)))
    c = T[0, -1, -1]
    A = T[1 0 0; 0 1 0]
    b = [one(T), Tirt2]
    G = SparseMatrixCSC(-one(T) * I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiNormInf{T, T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1 - Tirt2 atol=tol rtol=tol
    @test r.x ≈ [1, Tirt2, 1] atol=tol rtol=tol
    @test r.y ≈ [1, 1] atol=tol rtol=tol
end

function epinorminf2(T; options...)
    tol = 10 * sqrt(sqrt(eps(T)))
    l = 3
    L = 2l + 1
    c = collect(T, -l:l)
    A = spzeros(T, 2, L)
    A[1, 1] = A[1, L] = A[2, 1] = 1; A[2, L] = -1
    b = T[0, 0]
    G = [spzeros(T, 1, L); sparse(one(T) * I, L, L); spzeros(T, 1, L); sparse(T(2) * I, L, L)]
    h = zeros(T, 2L + 2); h[1] = 1; h[L + 2] = 1
    cones = CO.Cone{T}[CO.EpiNormInf{T, T}(L + 1, true), CO.EpiNormInf{T, T}(L + 1, false)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, obj_offset = one(T), options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -l + 2 atol=tol rtol=tol
    @test r.x[2] ≈ 0.5 atol=tol rtol=tol
    @test r.x[end - 1] ≈ -0.5 atol=tol rtol=tol
    @test sum(abs, r.x) ≈ 1 atol=tol rtol=tol
end

function epinorminf3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 0, 0, 0, 0, 0]
    A = zeros(T, 0, 6)
    b = zeros(T, 0)
    G = Diagonal(-one(T) * I, 6)
    h = zeros(T, 6)
    cones = CO.Cone{T}[CO.EpiNormInf{T, T}(6)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
end

function epinorminf4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 1, -1]
    A = T[1 0 0; 0 1 0]
    b = T[1, -0.4]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiNormInf{T, T}(3, true)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1 atol=tol rtol=tol
    @test r.x ≈ [1, -0.4, 0.6] atol=tol rtol=tol
    @test r.y ≈ [1, 0] atol=tol rtol=tol
end

function epinorminf5(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    c = T[1, 0, 0, 0, 0, 0]
    A = rand(T(-9):T(9), 3, 6)
    b = vec(sum(A, dims = 2))
    G = rand(T, 6, 6)
    h = vec(sum(G, dims = 2))
    cones = CO.Cone{T}[CO.EpiNormInf{T, T}(6, true)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 1 atol=tol rtol=tol
end

function epinormeucl1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Trt2 = sqrt(T(2))
    Tirt2 = inv(Trt2)
    c = T[0, -1, -1]
    A = T[10 0 0; 0 10 0]
    b = T[10, 10Tirt2]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiNormEucl{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -Trt2 atol=tol rtol=tol
    @test r.x ≈ [1, Tirt2, Tirt2] atol=tol rtol=tol
    @test r.y ≈ [Trt2 / 10, 0] atol=tol rtol=tol
end

function epinormeucl2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, -1, -1]
    A = T[1 0 0]
    b = T[0]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiNormEucl{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
end

function epinormeucl3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 0, 0]
    A = T[0 1 0]
    b = T[1]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiNormEucl{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 1 atol=tol rtol=tol
    @test r.x ≈ [1, 1, 0] atol=tol rtol=tol
end

function epipersquare1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 0, -1, -1]
    A = T[1 0 0 0; 0 1 0 0]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 4, 4)
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.EpiPerSquare{T}(4)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -sqrt(T(2)) atol=tol rtol=tol
    @test r.x[3:4] ≈ [1, 1] / sqrt(T(2)) atol=tol rtol=tol
end

function epipersquare2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Tirt2 = inv(sqrt(T(2)))
    c = T[0, 0, -1]
    A = T[1 0 0; 0 1 0]
    b = T[Tirt2 / 2, Tirt2]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiPerSquare{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, obj_offset = -one(T), options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -Tirt2 - 1 atol=tol rtol=tol
    @test r.x[2] ≈ Tirt2 atol=tol rtol=tol
end

function epipersquare3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 1, -1, -1]
    A = T[1 0 0 0]
    b = T[0]
    G = SparseMatrixCSC(-one(T) * I, 4, 4)
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.EpiPerSquare{T}(4)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, obj_offset = zero(T), options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
end

function possemideftri1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, -1, 0]
    A = T[1 0 0; 0 0 1]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)

    cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(3)]
    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -one(T) atol=tol rtol=tol
    @test r.x[2] ≈ one(T) atol=tol rtol=tol
end

function possemideftri2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, -1, 0]
    A = T[1 0 1]
    b = T[0]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)

    cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
end

# maximum eigenvalue problem
function possemideftri3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    c = T[1]
    A = zeros(T, 0, 1)
    b = T[]
    rand_mat = Hermitian(rand(T, 2, 2), :U)
    G = reshape(T[-1, 0, -1], 3, 1)
    h = -CO.smat_to_svec!(zeros(T, 3), rand_mat, rt2)
    cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    eig_max = maximum(eigvals(rand_mat))
    @test r.primal_obj ≈ eig_max atol=tol rtol=tol
    @test r.x[1] ≈ eig_max atol=tol rtol=tol
end

# dual formulation to the above
function possemideftri4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    s = 3
    rand_mat = Hermitian(rand(T, s, s), :U)
    dim = sum(1:s)
    c = -CO.smat_to_svec!(zeros(T, dim), rand_mat, rt2)
    A = reshape(CO.smat_to_svec!(zeros(T, dim), Matrix{T}(I, s, s), rt2), 1, dim)
    b = T[1]
    G = Diagonal(-one(T) * I, dim)
    h = zeros(T, dim)
    cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(dim)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    eig_max = maximum(eigvals(rand_mat))
    @test r.primal_obj ≈ -eig_max atol=tol rtol=tol
end

function possemideftricomplex1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Trt2 = sqrt(T(2))
    Trt2i = inv(Trt2)
    c = T[1, 0, 0, 1]
    A = T[0 0 1 0]
    b = T[1]
    G = Diagonal(-one(T) * I, 4)
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.PosSemidefTri{T, Complex{T}}(4)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ Trt2 atol=tol rtol=tol
    @test r.x ≈ [Trt2i, 0, 1, Trt2i] atol=tol rtol=tol
end

# maximum eigenvalue problem
function possemideftricomplex2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    c = T[1]
    A = zeros(T, 0, 1)
    b = T[]
    rand_mat = Hermitian(rand(Complex{T}, 2, 2), :U)
    G = reshape(T[-1, 0, 0, -1], 4, 1)
    h = -CO.smat_to_svec!(zeros(T, 4), rand_mat, rt2)
    cones = CO.Cone{T}[CO.PosSemidefTri{T, Complex{T}}(4)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    eig_max = maximum(eigvals(rand_mat))
    @test r.primal_obj ≈ eig_max atol=tol rtol=tol
    @test r.x[1] ≈ eig_max atol=tol rtol=tol
end

# dual formulation to the above
function possemideftricomplex3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    s = 3
    rand_mat = Hermitian(rand(Complex{T}, s, s), :U)
    dim = abs2(s)
    c = -CO.smat_to_svec!(zeros(T, dim), rand_mat, rt2)
    A = reshape(CO.smat_to_svec!(zeros(T, dim), Matrix{Complex{T}}(I, s, s), rt2), 1, dim)
    b = T[1]
    G = Diagonal(-one(T) * I, dim)
    h = zeros(T, dim)
    cones = CO.Cone{T}[CO.PosSemidefTri{T, Complex{T}}(dim)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    eig_max = maximum(eigvals(rand_mat))
    @test r.primal_obj ≈ -eig_max atol=tol rtol=tol
end

function epiperexp1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Texph = exp(T(0.5))
    c = T[1, 1, 1]
    A = T[0 1 0; 1 0 0]
    b = T[2, 1]
    G = sparse([3, 2, 1], [1, 2, 3], -ones(T, 3))
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiPerExp{T}()]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 2 * Texph + 3 atol=tol rtol=tol
    @test r.x ≈ [1, 2, 2 * Texph] atol=tol rtol=tol
    @test r.y ≈ -[1 + Texph / 2, 1 + Texph] atol=tol rtol=tol
    @test r.z ≈ -G * (c + A' * r.y) atol=tol rtol=tol
end

function epiperexp2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 0, -1]
    A = T[0 1 0]
    b = T[0]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiPerExp{T}()]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
end

function epiperexp3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 1, 1]
    A = zeros(T, 0, 3)
    b = zeros(T, 0)
    G = sparse([3, 2, 1, 4], [1, 2, 3, 1], -ones(T, 4))
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.EpiPerExp{T}(), CO.Nonnegative{T}(1)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
    @test isempty(r.y)
end

function epiperexp4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Texp2 = exp(T(-2))
    c = T[1, 0, 0]
    A = T[0 1 0; 0 0 1]
    b = T[1, -1]
    G = SparseMatrixCSC(-one(T) * I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.EpiPerExp{T}(true)]
    @warn "not sure if use_dual should work for EpiPerExp now" # TODO fix

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ Texp2 atol=tol rtol=tol
    @test r.x ≈ [Texp2, 1, -1] atol=tol rtol=tol
end

function epipersumexp1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    l = 5
    c = vcat(zero(T), -ones(T, l))
    A = hcat(one(T), zeros(T, 1, l))
    b = T[1]
    G = [-one(T) zeros(T, 1, l); zeros(T, 1, l + 1); zeros(T, l, 1) sparse(-one(T) * I, l, l)]
    h = zeros(T, l + 2)
    cones = CO.Cone{T}[CO.EpiPerSumExp{T}(l + 2)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.x[1] ≈ 1 atol=tol rtol=tol
    @test r.s[2] ≈ 0 atol=tol rtol=tol
    @test r.s[1] ≈ 1 atol=tol rtol=tol
end

function epipersumexp2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    l = 5
    c = vcat(zero(T), -ones(T, l))
    A = hcat(one(T), zeros(T, 1, l))
    b = T[1]
    G = [-one(T) spzeros(T, 1, l); spzeros(T, 1, l + 1); spzeros(T, l, 1) sparse(-one(T) * I, l, l)]
    h = zeros(T, l + 2); h[2] = 1
    cones = CO.Cone{T}[CO.EpiPerSumExp{T}(l + 2)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.x[1] ≈ 1 atol=tol rtol=tol
    @test r.s[2] ≈ 1 atol=tol rtol=tol
    @test r.s[2] * sum(exp, r.s[3:end] / r.s[2]) ≈ r.s[1] atol=tol rtol=tol
end

function hypopersumlog1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Texph = exp(T(0.5))
    c = T[1, 1, 1]
    A = T[0 1 0; 1 0 0]
    b = T[2, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.HypoPerSumLog{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 2 * Texph + 3 atol=tol rtol=tol
    @test r.x ≈ [1, 2, 2 * Texph] atol=tol rtol=tol
    @test r.y ≈ -[1 + Texph / 2, 1 + Texph] atol=tol rtol=tol
    @test r.z ≈ c + A' * r.y atol=tol rtol=tol
end

function hypopersumlog2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0, 0]
    A = T[0 1 0]
    b = T[0]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.HypoPerSumLog{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
end

function hypopersumlog3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 1, 1]
    A = zeros(T, 0, 3)
    b = zeros(T, 0)
    G = sparse([1, 2, 3, 4], [1, 2, 3, 1], -ones(T, 4))
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.HypoPerSumLog{T}(3), CO.Nonnegative{T}(1)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
    @test isempty(r.y)
end

function hypopersumlog4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Texp2 = exp(T(-2))
    c = T[0, 0, 1]
    A = T[0 1 0; 1 0 0]
    b = T[1, -1]
    G = SparseMatrixCSC(-one(T) * I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.HypoPerSumLog{T}(3, true)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ Texp2 atol=tol rtol=tol
    @test r.x ≈ [-1, 1, Texp2] atol=tol rtol=tol
end

function hypopersumlog5(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Tlogq = log(T(0.25))
    c = T[-1, 0, 0]
    A = T[0 1 1]
    b = T[1]
    G = sparse([1, 3, 4], [1, 2, 3], -ones(T, 3))
    h = T[0, 1, 0, 0]
    cones = CO.Cone{T}[CO.HypoPerSumLog{T}(4)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -Tlogq atol=tol rtol=tol
    @test r.x ≈ [Tlogq, 0.5, 0.5] atol=tol rtol=tol
    @test r.y ≈ [2] atol=tol rtol=tol
end

function hypopersumlog6(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0, 0]
    A = zeros(T, 0, 3)
    b = zeros(T, 0)
    G = sparse([1, 3, 4], [1, 2, 3], -ones(T, 3))
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.HypoPerSumLog{T}(4)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test r.x[1] ≈ 0 atol=tol rtol=tol
    @test isempty(r.y)
end

function hypogeomean1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0, 0]
    A = T[0 0 1; 0 1 0]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.HypoGeomean2{T}(ones(T, 2) / 2, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? 0 : -inv(sqrt(T(2)))) atol=tol rtol=tol
        @test r.x[2:3] ≈ [1, 0.5] atol=tol rtol=tol
    end
end

function hypogeomean2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    l = 4
    c = vcat(zero(T), ones(T, l))
    A = T[one(T) zeros(T, 1, l)]
    G = SparseMatrixCSC(-one(T) * I, l + 1, l + 1)
    h = zeros(T, l + 1)

    for is_dual in (true, false)
        b = is_dual ? [-one(T)] : [one(T)]
        cones = CO.Cone{T}[CO.HypoGeomean2{T}(fill(inv(T(l)), l), is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? 1 : l) atol=tol rtol=tol
        @test r.x[2:end] ≈ (is_dual ? fill(inv(T(l)), l) : ones(l)) atol=tol rtol=tol
    end
end

function hypogeomean3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    l = 4
    c = ones(T, l)
    A = zeros(T, 0, l)
    b = zeros(T, 0)
    G = [zeros(T, 1, l); Matrix{T}(-I, l, l)]
    h = zeros(T, l + 1)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.HypoGeomean2{T}(fill(inv(T(l)), l), is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function power1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 0, 1]
    A = T[1 0 0; 0 1 0]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)

    for is_dual in (false, true)
        cones = CO.Cone{T}[CO.Power{T}(ones(T, 2) / 2, 1, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? -sqrt(T(2)) : -inv(sqrt(T(2)))) atol=tol rtol=tol
        @test r.x[3] ≈ (is_dual ? -sqrt(T(2)) : -inv(sqrt(T(2)))) atol=tol rtol=tol
        @test r.x[1:2] ≈ [0.5, 1] atol=tol rtol=tol
    end
end

function power2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 0, -1, -1]
    A = T[0 1 0 0; 1 0 0 0]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 4, 4)
    h = zeros(T, 4)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.Power{T}(ones(T, 2) / 2, 2, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? -T(2) : -1) atol=tol rtol=tol
        @test norm(r.x[3:4]) ≈ (is_dual ? sqrt(T(2)) : inv(sqrt(T(2)))) atol=tol rtol=tol
        @test r.x[3:4] ≈ (is_dual ? ones(T, 2) : fill(inv(T(2)), 2)) atol=tol rtol=tol
        @test r.x[1:2] ≈ [1, 0.5] atol=tol rtol=tol
    end
end

function power3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    l = 4
    c = vcat(ones(T, l), zeros(T, 2))
    A = T[zeros(T, 1, l) one(T) zero(T); zeros(T, 1, l) zero(T) one(T)]
    G = SparseMatrixCSC(-one(T) * I, l + 2, l + 2)
    h = zeros(T, l + 2)

    for is_dual in (true, false)
        b = [one(T), zero(T)]
        cones = CO.Cone{T}[CO.Power{T}(fill(inv(T(l)), l), 2, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? 1 : T(l)) atol=tol rtol=tol
        @test r.x[1:l] ≈ (is_dual ? fill(inv(T(l)), l) : ones(l)) atol=tol rtol=tol
    end
end

function power4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    l = 4
    c = ones(T, l)
    A = zeros(T, 0, l)
    b = zeros(T, 0)
    G = [zeros(T, 3, l); Matrix{T}(-I, l, l)]
    h = zeros(T, l + 3)

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.Power{T}(fill(inv(T(l)), l), 3, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function epinormspectral1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (Xn, Xm) = (3, 4)
    Xnm = Xn * Xm
    c = vcat(one(T), zeros(T, Xnm))
    A = hcat(zeros(T, Xnm, 1), Matrix{T}(I, Xnm, Xnm))
    b = rand(T, Xnm)
    G = Matrix{T}(-I, Xnm + 1, Xnm + 1)
    h = vcat(zero(T), rand(T, Xnm))

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiNormSpectral{T}(Xn, Xm, is_dual)]

        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        if is_dual
            @test sum(svdvals(reshape(r.s[2:end], Xn, Xm))) ≈ r.s[1] atol=tol rtol=tol
            @test svdvals(reshape(r.z[2:end], Xn, Xm))[1] ≈ r.z[1] atol=tol rtol=tol
        else
            @test svdvals(reshape(r.s[2:end], Xn, Xm))[1] ≈ r.s[1] atol=tol rtol=tol
            @test sum(svdvals(reshape(r.z[2:end], Xn, Xm))) ≈ r.z[1] atol=tol rtol=tol
        end
    end
end

function epinormspectral2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    mat = T[2 3 4; 4 6 8]
    c = T[1, 0, 0, 0]
    A = zeros(T, 3, 4)
    A[1, 2] = 1
    A[2, 3] = 1
    A[3, 4] = 1
    b = T[4, 6, 4]
    G = zeros(T, 7, 4)
    G[1, 1] = G[3, 2] = G[5, 3] = G[6, 4] = -1
    h = T[0, 2, 0, 3, 0, 0, 8]

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiNormSpectral{T}(2, 3, is_dual)]
        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        if is_dual
            @test sum(svdvals(mat)) ≈ r.s[1] atol=tol rtol=tol
        else
            @test svdvals(mat)[1] ≈ r.s[1] atol=tol rtol=tol
        end
    end
end

# use definition of dual norm
function epinormspectral3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (Xn, Xm) = (3, 4)
    Xnm = Xn * Xm
    mat = rand(T, Xn, Xm)
    c = -vec(mat)
    A = zeros(T, 0, Xnm)
    b = T[]
    G = vcat(zeros(T, 1, Xnm), Matrix{T}(-I, Xnm, Xnm))
    h = vcat(one(T), zeros(T, Xnm))

    for is_dual in (true, false)
        cones = CO.Cone{T}[CO.EpiNormSpectral{T}(Xn, Xm, is_dual)]
        r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
        @test r.status == :Optimal
        if is_dual
            # the dual to the dual is the spectral norm
            @test r.primal_obj ≈ -svdvals(mat)[1] atol=tol rtol=tol
        else
            # the dual to the primal is the nuclear norm
            @test r.primal_obj ≈ -sum(svdvals(mat)) atol=tol rtol=tol
        end
    end
end

function hypoperlogdettri1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
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
    CO.smat_to_svec!(view(h, 3:dim), mat, rt2)
    cones = CO.Cone{T}[CO.HypoPerLogdetTri{T}(dim)]
    # unscale = [(i == j ? one(T) : inv(T(2))) for i in 1:side for j in 1:i]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.x[1] ≈ -r.primal_obj atol=tol rtol=tol
    @test r.x[2] ≈ 1 atol=tol rtol=tol
    sol_mat = zeros(T, side, side)
    CO.svec_to_smat!(sol_mat, r.s[3:end], rt2)
    @test r.s[2] * logdet(Symmetric(sol_mat, :U) / r.s[2]) ≈ r.s[1] atol=tol rtol=tol
    CO.svec_to_smat!(sol_mat, -r.z[3:end], rt2)
    @test r.z[1] * (logdet(Symmetric(sol_mat, :U) / r.z[1]) + T(side)) ≈ r.z[2] atol=tol rtol=tol
end

function hypoperlogdettri2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
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
    CO.smat_to_svec!(view(h, 3:dim), mat, rt2)
    cones = CO.Cone{T}[CO.HypoPerLogdetTri{T}(dim, true)]
    # unscale = [(i == j ? one(T) : inv(T(2))) for i in 1:side for j in 1:i]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.x[2] ≈ r.primal_obj atol=tol rtol=tol
    @test r.x[1] ≈ -1 atol=tol rtol=tol
    sol_mat = zeros(T, side, side)
    CO.svec_to_smat!(sol_mat, -r.s[3:end], rt2)
    @test r.s[1] * (logdet(Symmetric(sol_mat, :U) / r.s[1]) + T(side)) ≈ r.s[2] atol=tol rtol=tol
    CO.svec_to_smat!(sol_mat, r.z[3:end], rt2)
    @test r.z[2] * logdet(Symmetric(sol_mat, :U) / r.z[2]) ≈ r.z[1] atol=tol rtol=tol
end

function hypoperlogdettri3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
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
    CO.smat_to_svec!(view(h, 3:dim), mat, rt2)
    cones = CO.Cone{T}[CO.HypoPerLogdetTri{T}(dim)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :Optimal
    @test r.x[1] ≈ -r.primal_obj atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
end

function primalinfeas1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 0]
    A = T[1 1]
    b = [-T(2)]
    G = SparseMatrixCSC(-one(T) * I, 2, 2)
    h = zeros(T, 2)
    cones = CO.Cone{T}[CO.Nonnegative{T}(2)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :PrimalInfeasible
end

function primalinfeas2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 1, 1]
    A = zeros(T, 0, 3)
    b = T[]
    G = vcat(SparseMatrixCSC(-one(T) * I, 3, 3), Diagonal([one(T), one(T), -one(T)]))
    h = vcat(zeros(T, 3), one(T), one(T), -T(2))
    cones = CO.Cone{T}[CO.EpiNormEucl{T}(3), CO.Nonnegative{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :PrimalInfeasible
end

function primalinfeas3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = zeros(T, 3)
    A = SparseMatrixCSC(-one(T) * I, 3, 3)
    b = [one(T), one(T), T(3)]
    G = SparseMatrixCSC(-one(T) * I, 3, 3)
    h = zeros(T, 3)
    cones = CO.Cone{T}[CO.HypoPerSumLog{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :PrimalInfeasible
end

function dualinfeas1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, -1, 0]
    A = zeros(T, 0, 3)
    b = T[]
    G = repeat(SparseMatrixCSC(-one(T) * I, 3, 3), 2, 1)
    h = zeros(T, 6)
    cones = CO.Cone{T}[CO.EpiNormInf{T, T}(3), CO.EpiNormInf{T, T}(3, true)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :DualInfeasible
end

function dualinfeas2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0]
    A = zeros(T, 0, 2)
    b = T[]
    G = T[-1 0; 0 0; 0 -1]
    h = T[0, 1, 0]
    cones = CO.Cone{T}[CO.EpiPerSquare{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :DualInfeasible
end

function dualinfeas3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 1, 1, 0]
    A = zeros(T, 0, 4)
    b = T[]
    G = SparseMatrixCSC(-one(T) * I, 4, 4)
    h = zeros(T, 4)
    cones = CO.Cone{T}[CO.EpiPerSquare{T}(4)]

    r = build_solve_check(c, A, b, G, h, cones; atol = tol, options...)
    @test r.status == :DualInfeasible
end
