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
using SparseArrays
import GenericLinearAlgebra.svdvals
import GenericLinearAlgebra.eigvals
import DynamicPolynomials
import Hypatia
import Hypatia.ModelUtilities
import Hypatia.Cones
import Hypatia.Cones.Cone
import Hypatia.Solvers
const MU = Hypatia.ModelUtilities

# build model, solve, test conic certificates, and return solve information
function build_solve_check(
    c::Vector{T},
    A,
    b::Vector{T},
    G,
    h::Vector{T},
    cones::Vector{Cone{T}};
    obj_offset::T = zero(T),
    solver::Solvers.Solver{T} = Solvers.Solver{T}(),
    tol::Real = sqrt(eps(T)),
    ) where {T <: Real}
    model = Hypatia.Models.Model{T}(c, A, b, G, h, cones, obj_offset = obj_offset)

    Solvers.load(solver, model)
    Solvers.solve(solver)

    status = Solvers.get_status(solver)
    primal_obj = Solvers.get_primal_obj(solver)
    dual_obj = Solvers.get_dual_obj(solver)
    x = Solvers.get_x(solver)
    y = Solvers.get_y(solver)
    z = Solvers.get_z(solver)
    s = Solvers.get_s(solver)

    tol_sq = abs2(tol)
    tol_rt = sqrt(tol)
    if status == :Optimal
        @test primal_obj ≈ dual_obj atol=tol rtol=tol
        @test dot(c, x) + obj_offset ≈ primal_obj atol=tol_sq rtol=tol_sq
        @test -dot(b, y) - dot(h, z) + obj_offset ≈ dual_obj atol=tol_sq rtol=tol_sq
        @test A * x ≈ b atol=tol rtol=tol
        @test G * x + s ≈ h atol=tol rtol=tol
        @test G' * z + A' * y ≈ -c atol=tol rtol=tol
        @test dot(s, z) ≈ zero(T) atol=tol_rt rtol=tol_rt
    elseif status == :PrimalInfeasible
        @test dual_obj > obj_offset
        @test -dot(b, y) - dot(h, z) + obj_offset ≈ dual_obj atol=tol_sq rtol=tol_sq
        # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
        @test G' * z ≈ -A' * y atol=tol_rt rtol=tol_rt
    elseif status == :DualInfeasible
        @test primal_obj < obj_offset
        @test dot(c, x) + obj_offset ≈ primal_obj atol=tol_sq rtol=tol_sq
        # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
        @test G * x ≈ -s atol=tol_rt rtol=tol_rt
        @test A * x ≈ zeros(T, length(y)) atol=tol_rt rtol=tol_rt
    elseif status == :IllPosed
        # TODO primal vs dual ill-posed statuses and conditions
    end

    solve_time = Solvers.get_solve_time(solver)
    num_iters = Solvers.get_num_iters(solver)

    return (solver = solver, model = model, status = status,
        solve_time = solve_time, num_iters = num_iters,
        primal_obj = primal_obj, dual_obj = dual_obj,
        x = x, y = y, s = s, z = z)
end

function dimension1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0]
    A = zeros(T, 0, 2)
    b = T[]
    G = T[1 0]
    h = T[1]
    cones = Cone{T}[Cones.Nonnegative{T}(1)]

    for use_sparse in (false, true)
        if use_sparse
            A = sparse(A)
            G = sparse(G)
        end
        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
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
    c = zeros(T, n)
    A = rand(T(-9):T(9), p, n)
    G = Matrix{T}(10I, q, n)
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
    cones = Cone{T}[Cones.Nonnegative{T}(q)]

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

    @test_throws ErrorException options.linear_model{T}(c, A, b, G, h, Cone{T}[Cones.Nonnegative{T}(q)])
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

    @test_throws ErrorException options.linear_model{T}(c, A, b, G, h, Cone{T}[Cones.Nonnegative{T}(q)])
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
    cones = Cone{T}[Cones.Nonnegative{T}(q)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, obj_offset = one(T), options...)
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
    cones = Cone{T}[Cones.Nonnegative{T}(q)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
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
    cones = Cone{T}[Cones.Nonnegative{T}(q)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
end

function nonnegative4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-2, 0]
    A = zeros(T, 0, 2)
    b = zeros(T, 0)
    G = sparse([1, 1, 2, 3], [1, 2, 2, 2], T[1, -1, 1, -1], 3, 2)
    h = T[0, 2, 0]
    cones = Cone{T}[Cones.Nonnegative{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -4 atol=tol rtol=tol
    @test r.x ≈ [2, 2] atol=tol rtol=tol
    @test r.s ≈ [0, 0, 2] atol=tol rtol=tol
    @test r.z ≈ [2, 2, 0] atol=tol rtol=tol
end

function epinorminf1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Tirt2 = inv(sqrt(T(2)))
    c = T[0, -1, -1]
    A = T[1 0 0; 0 1 0]
    b = [one(T), Tirt2]
    G = SparseMatrixCSC(-one(T) * I, 3, 3)
    h = zeros(T, 3)
    cones = Cone{T}[Cones.EpiNormInf{T, T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
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
    cones = Cone{T}[Cones.EpiNormInf{T, T}(L + 1, use_dual = true), Cones.EpiNormInf{T, T}(L + 1, use_dual = false)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, obj_offset = one(T), options...)
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

    for use_dual in (true, false)
        cones = Cone{T}[Cones.EpiNormInf{T, T}(6, use_dual = use_dual)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function epinorminf4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 1, -1]
    A = T[1 0 0; 0 1 0]
    b = T[1, -0.4]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cones = Cone{T}[Cones.EpiNormInf{T, T}(3, use_dual = true)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
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
    cones = Cone{T}[Cones.EpiNormInf{T, T}(6, use_dual = true)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 1 atol=tol rtol=tol
end

function epinorminf6(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, -1, -1, -1, -1]
    A = T[1 0 0 0 0; 0 1 0 0 0; 0 0 0 1 0; 0 0 0 0 1]
    b = T[2, 0, 1, 0]
    G = SparseMatrixCSC(-one(T) * I, 5, 5)
    h = zeros(T, 5)
    cones = Cone{T}[Cones.EpiNormInf{T, Complex{T}}(5)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -3 atol=tol rtol=tol
    @test r.x ≈ [2, 0, 2, 1, 0] atol=tol rtol=tol
end

function epinorminf7(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 0, 0, 0, 0, 0, 0]
    A = zeros(T, 0, 7)
    b = zeros(T, 0)
    G = Diagonal(-one(T) * I, 7)
    h = zeros(T, 7)

    for use_dual in (true, false)
        cones = Cone{T}[Cones.EpiNormInf{T, Complex{T}}(7, use_dual = use_dual)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function epinorminf8(T; options...)
    tol = eps(T) ^ 0.2
    c = T[1, -1, 1, 1]
    A = T[1 0 0 0 ; 0 1 0 0; 0 0 1 0]
    b = T[-0.4, 0.3, -0.3]
    G = vcat(zeros(T, 1, 4), Diagonal(T[-1, -1, -1, -1]))
    h = T[1, 0, 0, 0, 0]
    cones = Cone{T}[Cones.EpiNormInf{T, Complex{T}}(5, use_dual = true)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1.4 atol=tol rtol=tol
    @test r.x ≈ [-0.4, 0.3, -0.3, -0.4] atol=tol rtol=tol
    @test r.y ≈ [0, 0.25, -0.25] atol=tol rtol=tol
    @test r.z ≈ [1.25, 1, -0.75, 0.75, 1] atol=tol rtol=tol
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
    cones = Cone{T}[Cones.EpiNormEucl{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
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
    cones = Cone{T}[Cones.EpiNormEucl{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
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
    cones = Cone{T}[Cones.EpiNormEucl{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
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
    cones = Cone{T}[Cones.EpiPerSquare{T}(4)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
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
    cones = Cone{T}[Cones.EpiPerSquare{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, obj_offset = -one(T), options...)
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
    cones = Cone{T}[Cones.EpiPerSquare{T}(4)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
end

function epipersquare4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = zeros(T, 7)
    c[1] = -1
    A = zeros(T, 0, 7)
    b = zeros(T, 0)
    G = sparse(
        [1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 7, 2, 3, 4, 2, 3, 5, 4, 7, 6, 5, 6, 7],
        T[1, -0.5, 1, 1, 1, -1, -1, -1, -1, -0.5, -1, -1, -1, -1],
        11, 7)
    h = zeros(T, 11)
    h[2] = 3
    cones = Cone{T}[Cones.Nonnegative{T}(2), Cones.EpiPerSquare{T}(3), Cones.EpiPerSquare{T}(3), Cones.EpiPerSquare{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1 atol=tol rtol=tol
    rt2 = sqrt(T(2))
    @test r.x ≈ [1, 1, 1, 1, rt2, rt2, 2] atol=tol rtol=tol
    @test r.s ≈ [0, 0, 1, 1, rt2, 1, 1, rt2, rt2, rt2, 2] atol=tol rtol=tol
    inv3 = inv(T(3))
    rt2inv3 = inv3 * rt2
    @test r.z ≈ [1, inv3, inv3, inv3, -rt2inv3, inv3, inv3, -rt2inv3, rt2inv3, rt2inv3, -2 * inv3] atol=tol rtol=tol
end

# TODO add use_dual = true tests
function episumperentropy1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    for w_dim in [1, 2, 3]
        dim = 1 + 2 * w_dim
        c = T[1]
        A = zeros(T, 0, 1)
        b = zeros(T, 0)
        G = zeros(T, dim, 1)
        G[1, 1] = -1
        h = zeros(T, dim)
        h[2:(w_dim + 1)] .= 1
        w = rand(T, w_dim) .+ 1
        h[(w_dim + 2):end] .= w
        cones = Cone{T}[Cones.EpiSumPerEntropy{T}(dim)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ sum(wi * log(wi) for wi in w) atol=tol rtol=tol
    end
end

function episumperentropy2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    for w_dim in [1, 2, 4]
        dim = 1 + 2 * w_dim
        c = fill(-one(T), w_dim)
        A = zeros(T, 0, w_dim)
        b = zeros(T, 0)
        G = vcat(zeros(T, 1 + w_dim, w_dim), Matrix{T}(-I, w_dim, w_dim))
        h = vcat(zero(T), ones(T, w_dim), zeros(T, w_dim))
        cones = Cone{T}[Cones.EpiSumPerEntropy{T}(dim)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -w_dim atol=tol rtol=tol
        @test r.x ≈ fill(1, w_dim) atol=tol rtol=tol
    end
end

function episumperentropy3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    for w_dim in [2, 4]
        dim = 1 + 2 * w_dim
        c = fill(-one(T), w_dim)
        A = ones(T, 1, w_dim)
        b = T[dim]
        G = vcat(zeros(T, 1, w_dim), Matrix{T}(-I, w_dim, w_dim), zeros(T, w_dim, w_dim))
        h = vcat(zeros(T, 1 + w_dim), ones(T, w_dim))
        cones = Cone{T}[Cones.EpiSumPerEntropy{T}(dim)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -dim atol=tol rtol=tol
        @test r.x ≈ fill(dim / w_dim, w_dim) atol=tol rtol=tol
    end
end

function episumperentropy4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1]
    A = zeros(T, 0, 1)
    b = zeros(T, 0)
    G = Matrix{T}(-I, 5, 1)
    h = T[0, 1, 5, 2, 3]
    cones = Cone{T}[Cones.EpiSumPerEntropy{T}(5)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    entr = 2 * log(T(2)) + 3 * log(3 / T(5))
    @test r.primal_obj ≈ entr atol=tol rtol=tol
    @test r.s ≈ [entr, 1, 5, 2, 3] atol=tol rtol=tol
    @test r.z ≈ [1, 2, 3 / T(5), log(inv(T(2))) - 1, log(5 / T(3)) - 1] atol=tol rtol=tol
end

function episumperentropy5(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, -1]
    A = zeros(T, 0, 2)
    b = zeros(T, 0)
    G = vcat(zeros(T, 4, 2), fill(-one(T), 3, 2), [-1, 0]')
    h = T[0, 1, 1, 1, 0, 0, 0, 0]
    cones = Cone{T}[Cones.EpiSumPerEntropy{T}(7), Cones.Nonnegative{T}(1)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1 atol=tol rtol=tol
    @test r.s ≈ [0, 1, 1, 1, 1, 1, 1, 0] atol=tol rtol=tol
    @test r.z ≈ inv(T(3)) * [1, 1, 1, 1, -1, -1, -1, 3] atol=tol rtol=tol
end

function episumperentropy6(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0, 0, 0, 0]
    A = zeros(T, 0, 5)
    b = zeros(T, 0)
    G = sparse(
        [2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 9, 9, 9],
        [2, 3, 4, 1, 1, 1, 5, 5, 5, 5, 2, 3, 4],
        T[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1],
        9, 5)
    h = vcat(zeros(T, 8), T(3))
    cones = Cone{T}[Cones.EpiSumPerEntropy{T}(7), Cones.Nonnegative{T}(2)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1 atol=tol rtol=tol
    @test r.s ≈ [0, 1, 1, 1, 1, 1, 1, 0, 0] atol=tol rtol=tol
    @test r.z ≈ inv(T(3)) * [1, 1, 1, 1, -1, -1, -1, 3, 1] atol=tol rtol=tol
end

function hypoperlog1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Texph = exp(T(0.5))
    c = T[1, 1, 1]
    A = T[0 1 0; 1 0 0]
    b = T[2, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)
    cones = Cone{T}[Cones.HypoPerLog{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 2 * Texph + 3 atol=tol rtol=tol
    @test r.x ≈ [1, 2, 2 * Texph] atol=tol rtol=tol
    @test r.y ≈ -[1 + Texph / 2, 1 + Texph] atol=tol rtol=tol
    @test r.z ≈ c + A' * r.y atol=tol rtol=tol
end

function hypoperlog2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0, 0]
    A = T[0 1 0]
    b = T[0]
    G = Diagonal(-one(T) * I, 3)
    h = zeros(T, 3)
    cones = Cone{T}[Cones.HypoPerLog{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
end

function hypoperlog3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 1, 1]
    A = zeros(T, 0, 3)
    b = zeros(T, 0)
    G = sparse([1, 2, 3, 4], [1, 2, 3, 1], -ones(T, 4))
    h = zeros(T, 4)
    cones = Cone{T}[Cones.HypoPerLog{T}(3), Cones.Nonnegative{T}(1)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
    @test isempty(r.y)
end

function hypoperlog4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Texp2 = exp(T(-2))
    c = T[0, 0, 1]
    A = T[0 1 0; 1 0 0]
    b = T[1, -1]
    G = SparseMatrixCSC(-one(T) * I, 3, 3)
    h = zeros(T, 3)
    cones = Cone{T}[Cones.HypoPerLog{T}(3, use_dual = true)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ Texp2 atol=tol rtol=tol
    @test r.x ≈ [-1, 1, Texp2] atol=tol rtol=tol
end

function hypoperlog5(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Tlogq = log(T(0.25))
    c = T[-1, 0, 0]
    A = T[0 1 1]
    b = T[1]
    G = sparse([1, 3, 4], [1, 2, 3], -ones(T, 3))
    h = T[0, 1, 0, 0]
    cones = Cone{T}[Cones.HypoPerLog{T}(4)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -Tlogq atol=tol rtol=tol
    @test r.x ≈ [Tlogq, 0.5, 0.5] atol=tol rtol=tol
    @test r.y ≈ [2] atol=tol rtol=tol
end

function hypoperlog6(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0, 0]
    A = zeros(T, 0, 3)
    b = zeros(T, 0)
    G = sparse([1, 3, 4], [1, 2, 3], -ones(T, 3))
    h = zeros(T, 4)
    cones = Cone{T}[Cones.HypoPerLog{T}(4)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test r.x[1] ≈ 0 atol=tol rtol=tol
    @test isempty(r.y)
end

function hypoperlog7(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = zeros(T, 4)
    c[1] = -2
    A = zeros(T, 0, 4)
    b = zeros(T, 0)
    G = sparse(
        [1, 2, 2, 3, 4, 5, 6],
        [2, 1, 3, 4, 4, 3, 2],
        T[1, 1, -1, -1, -1, -1, -1],
        6, 4)
    h = zeros(T, 6)
    h[1] = 2
    cones = Cone{T}[Cones.Nonnegative{T}(3), Cones.HypoPerLog{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -4 atol=tol rtol=tol
    @test r.x ≈ [2, 2, 2, 0] atol=tol rtol=tol
    @test r.s ≈ [0, 0, 0, 0, 2, 2] atol=tol rtol=tol
    @test r.z ≈ [2, 2, 2, -2, -2, 2] atol=tol rtol=tol
end

function hypogeomean1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0, 0]
    A = T[0 0 1; 0 1 0]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)

    for use_dual in (true, false)
        cones = Cone{T}[Cones.HypoGeomean{T}(ones(T, 2) / 2, use_dual = use_dual)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (use_dual ? 0 : -inv(sqrt(T(2)))) atol=tol rtol=tol
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

    for use_dual in (true, false)
        b = use_dual ? [-one(T)] : [one(T)]
        cones = Cone{T}[Cones.HypoGeomean{T}(fill(inv(T(l)), l), use_dual = use_dual)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (use_dual ? 1 : l) atol=tol rtol=tol
        @test r.x[2:end] ≈ (use_dual ? fill(inv(T(l)), l) : ones(l)) atol=tol rtol=tol
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

    for use_dual in (true, false)
        cones = Cone{T}[Cones.HypoGeomean{T}(fill(inv(T(l)), l), use_dual = use_dual)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function hypogeomean4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0, 0, 0]
    A = zeros(T, 0, 4)
    b = zeros(T, 0)
    G = vcat(Matrix{T}(-I, 4, 4), T[0, 1, 1, 1]')
    h = T[0, 0, 0, 0, 3]
    cones = Cone{T}[Cones.HypoGeomean{T}(fill(inv(T(3)), 3)), Cones.Nonnegative{T}(1)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1 atol=tol rtol=tol
    @test r.x ≈ [1, 1, 1, 1] atol=tol rtol=tol
    @test r.s ≈ [1, 1, 1, 1, 0] atol=tol rtol=tol
    @test r.z ≈ vcat(-1, fill(inv(T(3)), 4)) atol=tol rtol=tol
end

function hypogeomean5(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-2, 0]
    A = zeros(T, 0, 2)
    b = zeros(T, 0)
    G = sparse([1, 2, 3], [1, 2, 2], T[-1, -1, 1], 3, 2)
    h = T[0, 0, 2]
    cones = Cone{T}[Cones.HypoGeomean{T}([one(T)]), Cones.Nonnegative{T}(1)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -4 atol=tol rtol=tol
    @test r.x ≈ [2, 2] atol=tol rtol=tol
    @test r.s ≈ [2, 2, 0] atol=tol rtol=tol
    @test r.z ≈ [-2, 2, 2] atol=tol rtol=tol
end

function hypogeomean6(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = zeros(T, 10)
    c[1] = -1
    A = hcat(zeros(T, 9), Matrix{T}(I, 9, 9))
    b = ones(T, 9)
    G = Matrix{T}(-I, 10, 10)
    h = zeros(T, 10)
    cones = Cone{T}[Cones.HypoGeomean{T}(fill(inv(T(9)), 9))]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1 atol=tol rtol=tol
    @test r.x ≈ ones(T, 10) atol=tol rtol=tol
    @test r.z ≈ vcat(-one(T), fill(inv(T(9)), 9)) atol=tol rtol=tol
    @test r.y ≈ fill(inv(T(9)), 9) atol=tol rtol=tol
end

function power1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 0, 1]
    A = T[1 0 0; 0 1 0]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)

    for use_dual in (false,)# true)
        cones = Cone{T}[Cones.Power{T}(ones(T, 2) / 2, 1, use_dual = use_dual)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (use_dual ? -sqrt(T(2)) : -inv(sqrt(T(2)))) atol=tol rtol=tol
        @test r.x[3] ≈ (use_dual ? -sqrt(T(2)) : -inv(sqrt(T(2)))) atol=tol rtol=tol
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

    for use_dual in (false,)# true)
        cones = Cone{T}[Cones.Power{T}(ones(T, 2) / 2, 2, use_dual = use_dual)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (use_dual ? -T(2) : -1) atol=tol rtol=tol
        @test norm(r.x[3:4]) ≈ (use_dual ? sqrt(T(2)) : inv(sqrt(T(2)))) atol=tol rtol=tol
        @test r.x[3:4] ≈ (use_dual ? ones(T, 2) : fill(inv(T(2)), 2)) atol=tol rtol=tol
        @test r.x[1:2] ≈ [1, 0.5] atol=tol rtol=tol
    end
end

function power3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    l = 4
    c = vcat(fill(T(10), l), zeros(T, 2))
    A = T[zeros(T, 1, l) one(T) zero(T); zeros(T, 1, l) zero(T) one(T)]
    G = SparseMatrixCSC(-T(10) * I, l + 2, l + 2)
    h = zeros(T, l + 2)

    for use_dual in (false,)# true)
        b = [one(T), zero(T)]
        cones = Cone{T}[Cones.Power{T}(fill(inv(T(l)), l), 2, use_dual = use_dual)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (use_dual ? 10 : 10 * T(l)) atol=tol rtol=tol
        @test r.x[1:l] ≈ (use_dual ? fill(inv(T(l)), l) : ones(l)) atol=tol rtol=tol
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

    for use_dual in (false,)# true)
        cones = Cone{T}[Cones.Power{T}(fill(inv(T(l)), l), 3, use_dual = use_dual)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function epinormspectral1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (Xn, Xm) = (3, 4)
    for is_complex in (false, true)
        dim = Xn * Xm
        if is_complex
            dim *= 2
        end
        c = vcat(one(T), zeros(T, dim))
        A = hcat(zeros(T, dim, 1), Matrix{T}(I, dim, dim))
        b = rand(T, dim)
        G = Matrix{T}(-I, dim + 1, dim + 1)
        h = vcat(zero(T), rand(T, dim))

        for use_dual in (true, false)
            R = (is_complex ? Complex{T} : T)
            cones = Cone{T}[Cones.EpiNormSpectral{T, R}(Xn, Xm, use_dual = use_dual)]

            r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
            @test r.status == :Optimal

            S = zeros(R, Xn, Xm)
            @views Cones.vec_copy_to!(S, r.s[2:end])
            prim_svdvals = svdvals(S)
            Z = similar(S)
            @views Cones.vec_copy_to!(Z, r.z[2:end])
            dual_svdvals = svdvals(Z)
            if use_dual
                @test sum(prim_svdvals) ≈ r.s[1] atol=tol rtol=tol
                @test dual_svdvals[1] ≈ r.z[1] atol=tol rtol=tol
            else
                @test prim_svdvals[1] ≈ r.s[1] atol=tol rtol=tol
                @test sum(dual_svdvals) ≈ r.z[1] atol=tol rtol=tol
            end
        end
    end
end

function epinormspectral2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (Xn, Xm) = (3, 4)
    for is_complex in (false, true)
        R = (is_complex ? Complex{T} : T)
        dim = Xn * Xm
        if is_complex
            dim *= 2
        end
        mat = rand(R, Xn, Xm)
        c = zeros(T, dim)
        Cones.vec_copy_to!(c, mat)
        c .*= -1
        A = zeros(T, 0, dim)
        b = T[]
        G = vcat(zeros(T, 1, dim), Matrix{T}(-I, dim, dim))
        h = vcat(one(T), zeros(T, dim))

        for use_dual in (true, false)
            cones = Cone{T}[Cones.EpiNormSpectral{T, R}(Xn, Xm, use_dual = use_dual)]
            r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
            @test r.status == :Optimal
            if use_dual
                @test r.primal_obj ≈ -svdvals(mat)[1] atol=tol rtol=tol
            else
                @test r.primal_obj ≈ -sum(svdvals(mat)) atol=tol rtol=tol
            end
        end
    end
end

function epinormspectral3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    for is_complex in (false, true), (Xn, Xm) in ((1, 1), (1, 3), (2, 2), (3, 4))
        dim = Xn * Xm
        if is_complex
            dim *= 2
        end
        c = fill(-one(T), dim)
        A = zeros(T, 0, dim)
        b = T[]
        G = vcat(zeros(T, 1, dim), Matrix{T}(-I, dim, dim))
        h = zeros(T, dim + 1)

        for use_dual in (true, false)
            R = (is_complex ? Complex{T} : T)
            cones = Cone{T}[Cones.EpiNormSpectral{T, R}(Xn, Xm, use_dual = use_dual)]
            r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
            @test r.status == :Optimal
            @test r.primal_obj ≈ 0 atol=tol rtol=tol
            @test norm(r.x) ≈ 0 atol=tol rtol=tol
        end
    end
end

function epinormspectral4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1]
    A = zeros(T, 0, 1)
    b = T[]
    G = zeros(T, 7, 1)
    G[1, 1] = -1
    h = T[0, 1, 1, 1, -1, 0, 1]

    rt2 = sqrt(T(2))
    rt3 = sqrt(T(3))
    invrt2 = inv(rt2)
    invrt3 = inv(rt3)
    for use_dual in (true, false)
        cones = Cone{T}[Cones.EpiNormSpectral{T, T}(2, 3, use_dual = use_dual)]
        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        if use_dual
            @test r.primal_obj ≈ rt2 + rt3 atol=tol rtol=tol
            @test r.s ≈ T[rt2 + rt3, 1, 1, 1, -1, 0, 1] atol=tol rtol=tol
            @test r.z ≈ T[1, -invrt2, -invrt3, -invrt2, invrt3, 0, -invrt3] atol=tol rtol=tol
        else
            @test r.primal_obj ≈ rt3 atol=tol rtol=tol
            @test r.s ≈ T[rt3, 1, 1, 1, -1, 0, 1] atol=tol rtol=tol
            @test r.z ≈ T[1, 0, -invrt3, 0, invrt3, 0, -invrt3] atol=tol rtol=tol
        end
    end
end

function matrixepipersquare1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    for is_complex in (false, true), (Xn, Xm) in [(1, 1), (1, 3), (2, 2), (2, 3)]
        R = (is_complex ? Complex{T} : T)
        per_idx = (is_complex ? Xn ^ 2 + 1 : Cones.svec_length(Xn) + 1)
        dim = per_idx + (is_complex ? 2 : 1) * Xn * Xm
        c = T[1]
        A = zeros(T, 0, 1)
        b = T[]
        G = zeros(T, dim, 1)
        G[per_idx] = -1
        h = zeros(T, dim)
        @views Cones.smat_to_svec!(h[1:(per_idx - 1)], Matrix{R}(I, Xn, Xn), sqrt(T(2)))
        W = rand(R, Xn, Xm)
        @views Cones.vec_copy_to!(h[(per_idx + 1):end], W)
        WWt = Hermitian(W * W')
        dual_epi = tr(WWt) / 2
        primal_epi = svdvals(WWt)[1] / 2

        for use_dual in (false, true)
            cones = Cone{T}[Cones.MatrixEpiPerSquare{T, R}(Xn, Xm, use_dual = use_dual)]
            r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
            @test r.status == :Optimal
            @test r.primal_obj >= 0
            if use_dual
                @test r.primal_obj ≈ dual_epi atol=tol rtol=tol
                @test r.s[per_idx] ≈ dual_epi atol=tol rtol=tol
                @test r.z[per_idx] ≈ 1 atol=tol rtol=tol
            else
                @test r.primal_obj ≈ primal_epi atol=tol rtol=tol
                @test r.s[per_idx] ≈ primal_epi atol=tol rtol=tol
                @test r.z[per_idx] ≈ 1 atol=tol rtol=tol
            end
        end
    end
end

function matrixepipersquare2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (Xn, Xm) = (3, 4)
    for is_complex in (false, true)
        R = (is_complex ? Complex{T} : T)
        per_idx = (is_complex ? Xn ^ 2 + 1 : Cones.svec_length(Xn) + 1)
        dim = per_idx + (is_complex ? 2 : 1) * Xn * Xm
        c = T[1]
        A = zeros(T, 0, 1)
        b = T[]
        G = zeros(T, dim, 1)
        G[per_idx] = -1
        h = zeros(T, dim)
        U_half = rand(R, Xn, Xn)
        U = Hermitian(U_half * U_half')
        @views Cones.smat_to_svec!(h[1:(per_idx - 1)], U.data, sqrt(T(2)))
        W = rand(R, Xn, Xm)
        @views Cones.vec_copy_to!(h[(per_idx + 1):end], W)

        for use_dual in (false, true)
            cones = Cone{T}[Cones.MatrixEpiPerSquare{T, R}(Xn, Xm, use_dual = use_dual)]
            r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
            @test r.status == :Optimal
            @test r.primal_obj >= 0
            if use_dual
                @test 2 * r.s[per_idx] ≈ tr(W' * (U \ W)) atol=tol rtol=tol
            else
                primal_viol = Hermitian(2 * r.s[per_idx] * U - W * W')
                @test minimum(eigvals(primal_viol)) ≈ 0 atol=tol rtol=tol
            end
        end
    end
end

function matrixepipersquare3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    (Xn, Xm) = (2, 4)
    for is_complex in (false, true)
        R = (is_complex ? Complex{T} : T)
        per_idx = (is_complex ? Xn ^ 2 + 1 : Cones.svec_length(Xn) + 1)
        W_dim = (is_complex ? 2 : 1) * Xn * Xm
        dim = per_idx + W_dim
        c = ones(T, W_dim)
        A = zeros(T, 0, W_dim)
        b = T[]
        G = vcat(zeros(T, per_idx, W_dim), Matrix{T}(-10I, W_dim, W_dim))
        h = zeros(T, dim)
        U_half = rand(R, Xn, Xn)
        U = Hermitian(U_half * U_half')
        @views Cones.smat_to_svec!(h[1:(per_idx - 1)], U.data, sqrt(T(2)))

        for use_dual in (false, true)
            cones = Cone{T}[Cones.MatrixEpiPerSquare{T, R}(Xn, Xm, use_dual = use_dual)]
            r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
            @test r.status == :Optimal
            @test norm(r.x) ≈ 0 atol=2tol rtol=2tol
        end
    end
end

function linmatrixineq1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    for side in [2, 3, 5], R in [T, Complex{T}]
        c = T[1]
        A = zeros(T, 0, 1)
        b = T[]
        G = zeros(T, 2, 1)
        G[1, 1] = -1
        h = T[0, 2]
        A_1_half = rand(R, side, side)
        A_1 = Hermitian(A_1_half * A_1_half' + 2I)
        F = eigen(A_1)
        val_1 = F.values[end]
        vec_1 = F.vectors[:, end]
        As = [A_1, Hermitian(-vec_1 * vec_1')]
        cones = Cone{T}[Cones.LinMatrixIneq{T}(As)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 2 / val_1 atol=tol rtol=tol
        @test r.s ≈ [2 / val_1, 2] atol=tol rtol=tol
    end
end

function linmatrixineq2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Random.seed!(1)
    for Rs in [[T, T], [Complex{T}, Complex{T}], [T, Complex{T}, T], [Complex{T}, T, T]]
        dim = length(Rs)
        c = ones(T, dim - 1)
        A = zeros(T, 0, dim - 1)
        b = T[]
        G = vcat(spzeros(T, 1, dim - 1), sparse(-one(T) * I, dim - 1, dim - 1))
        h = zeros(T, dim)
        h[1] = 1
        As = Hermitian[]
        for R in Rs
            A_half = rand(R, 3, 3)
            push!(As, Hermitian(A_half * A_half'))
        end
        As[1] += I
        cones = Cone{T}[Cones.LinMatrixIneq{T}(As)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj < 0
    end
end

function linmatrixineq3(T; options...)
    dense1 = [1 0; 0 1]
    dense2 = [1 0; 0 -1]
    sparse1 = sparse(dense1)
    sparse2 = sparse(dense2)
    diag1 = Diagonal([1, 1])
    diag2 = Diagonal([1, -1])
    # NOTE not all combinations work due to missing methods in LinearAlgebra
    As_list = [
        [dense1, dense2],
        # [dense1, sparse2],
        [dense1, diag2],
        # [sparse1, dense2],
        [sparse1, sparse2],
        # [sparse1, diag2],
        [diag1, dense2],
        # [diag1, sparse2],
        [diag1, diag2],
        [I, dense2],
        # [I, sparse2],
        [I, diag2],
        ]
    for As in As_list
        tol = sqrt(sqrt(eps(T)))
        c = T[1]
        A = zeros(T, 0, 1)
        b = T[]
        G = zeros(T, 2, 1)
        G[1, 1] = -1
        h = T[0, -1]
        cones = Cone{T}[Cones.LinMatrixIneq{T}(As)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 1 atol=tol rtol=tol
        @test r.s ≈ [1, -1] atol=tol rtol=tol
    end
end

function possemideftri1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, -1, 0]
    A = T[1 0 0; 0 0 1]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)
    cones = Cone{T}[Cones.PosSemidefTri{T, T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
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
    cones = Cone{T}[Cones.PosSemidefTri{T, T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test norm(r.x) ≈ 0 atol=tol rtol=tol
end

function possemideftri3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    c = T[1]
    A = zeros(T, 0, 1)
    b = T[]
    rand_mat = Hermitian(rand(T, 2, 2), :U)
    G = reshape(T[-1, 0, -1], 3, 1)
    h = -Cones.smat_to_svec!(zeros(T, 3), rand_mat, rt2)
    cones = Cone{T}[Cones.PosSemidefTri{T, T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    eig_max = maximum(eigvals(rand_mat))
    @test r.primal_obj ≈ eig_max atol=tol rtol=tol
    @test r.x[1] ≈ eig_max atol=tol rtol=tol
end

function possemideftri4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    s = 3
    rand_mat = Hermitian(rand(T, s, s), :U)
    dim = sum(1:s)
    c = -Cones.smat_to_svec!(zeros(T, dim), rand_mat, rt2)
    A = reshape(Cones.smat_to_svec!(zeros(T, dim), Matrix{T}(I, s, s), rt2), 1, dim)
    b = T[1]
    G = Diagonal(-one(T) * I, dim)
    h = zeros(T, dim)
    cones = Cone{T}[Cones.PosSemidefTri{T, T}(dim)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    eig_max = maximum(eigvals(rand_mat))
    @test r.primal_obj ≈ -eig_max atol=tol rtol=tol
end

function possemideftri5(T; options...)
    tol = sqrt(sqrt(eps(T)))
    Trt2 = sqrt(T(2))
    Trt2i = inv(Trt2)
    c = T[1, 0, 0, 1]
    A = T[0 0 1 0]
    b = T[1]
    G = Diagonal(-one(T) * I, 4)
    h = zeros(T, 4)
    cones = Cone{T}[Cones.PosSemidefTri{T, Complex{T}}(4)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ Trt2 atol=tol rtol=tol
    @test r.x ≈ [Trt2i, 0, 1, Trt2i] atol=tol rtol=tol
end

function possemideftri6(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    c = T[1]
    A = zeros(T, 0, 1)
    b = T[]
    rand_mat = Hermitian(rand(Complex{T}, 2, 2), :U)
    G = reshape(T[-1, 0, 0, -1], 4, 1)
    h = -Cones.smat_to_svec!(zeros(T, 4), rand_mat, rt2)
    cones = Cone{T}[Cones.PosSemidefTri{T, Complex{T}}(4)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    eig_max = maximum(eigvals(rand_mat))
    @test r.primal_obj ≈ eig_max atol=tol rtol=tol
    @test r.x[1] ≈ eig_max atol=tol rtol=tol
end

function possemideftri7(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 3
    rand_mat = Hermitian(rand(Complex{T}, side, side), :U)
    dim = abs2(side)
    c = -Cones.smat_to_svec!(zeros(T, dim), rand_mat, rt2)
    A = reshape(Cones.smat_to_svec!(zeros(T, dim), Matrix{Complex{T}}(I, side, side), rt2), 1, dim)
    b = T[1]
    G = Diagonal(-one(T) * I, dim)
    h = zeros(T, dim)
    cones = Cone{T}[Cones.PosSemidefTri{T, Complex{T}}(dim)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    eig_max = maximum(eigvals(rand_mat))
    @test r.primal_obj ≈ -eig_max atol=tol rtol=tol
end

function possemideftri8(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    c = T[1]
    A = zeros(T, 0, 1)
    b = T[]
    G = zeros(T, 15, 1)
    G[[1, 3, 6, 10, 15]] .= -1
    h = zeros(T, 15)
    @. h[[7, 8, 9, 11, 12, 13]] = rt2 * [1, 1, 0, 1, -1, 1]
    cones = Cone{T}[Cones.PosSemidefTri{T, T}(15)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    rt3 = sqrt(T(3))
    @test r.primal_obj ≈ rt3 atol=tol rtol=tol
    @test r.s ≈ [rt3, 0, rt3, 0, 0, rt3, rt2, rt2, 0, rt3, rt2, -rt2, rt2, 0, rt3] atol=tol rtol=tol
    inv6 = inv(T(6))
    rt2inv6 = rt2 / 6
    invrt6 = inv(rt2 * rt3)
    @test r.z ≈ [inv6, -rt2inv6, inv6, rt2inv6, -rt2inv6, inv6, 0, 0, 0, 0, -invrt6, invrt6, -invrt6, 0, inv(T(2))] atol=tol rtol=tol
end

function possemideftri9(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    inv2 = inv(T(2))
    c = vcat(one(T), zeros(T, 9))
    A = zeros(T, 0, 10)
    b = T[]
    G = zeros(T, 16, 10)
    G[1, 2] = G[1, 4] = G[1, 7] = G[1, 8] = G[1, 10] = inv2
    G[1, 1] = G[2, 2] = G[4, 4] = G[7, 7] = G[11, 8] = G[16, 10] = -1
    G[3, 3] = G[5, 5] = G[6, 6] = G[15, 9] = -rt2
    h = zeros(T, 16)
    @. h[[8, 9, 10, 12, 13, 14]] = rt2 * [1, 1, 0, 1, -1, 1]
    cones = Cone{T}[Cones.Nonnegative{T}(1), Cones.PosSemidefTri{T, T}(15)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    rt3 = sqrt(T(3))
    @test r.primal_obj ≈ rt2 + rt3 atol=tol rtol=tol
    invrt2 = inv(rt2)
    invrt3 = inv(rt3)
    @test r.s ≈ [0, invrt2 + invrt3, 1 - rt2 / rt3, invrt2 + invrt3, rt2 * invrt3, -rt2 * invrt3, invrt3, rt2, rt2, 0, rt2, rt2, -rt2, rt2, 0, rt3] atol=tol rtol=tol
    invrt6 = invrt2 * invrt3
    @test r.z ≈ [1, inv2, 0, inv2, 0, 0, inv2, -inv2, -inv2, 0, inv2, -invrt6, invrt6, -invrt6, 0, inv2] atol=tol rtol=tol
end

function possemideftrisparse1(T; options...)
    if !(T <: LinearAlgebra.BlasReal)
        return # only works with BLAS real types
    end
    tol = sqrt(sqrt(eps(T)))
    c = T[0, -1, 0]
    A = T[1 0 0; 0 0 1]
    b = T[0.5, 1]
    G = Matrix{T}(-I, 3, 3)
    h = zeros(T, 3)
    row_idxs = [1, 2, 2]
    col_idxs = [1, 1, 2]
    cones = Cone{T}[Cones.PosSemidefTriSparse{T, T}(2, row_idxs, col_idxs)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -one(T) atol=tol rtol=tol
    @test r.x[2] ≈ one(T) atol=tol rtol=tol
end

function possemideftrisparse2(T; options...)
    if !(T <: LinearAlgebra.BlasReal)
        return # only works with BLAS real types
    end
    tol = sqrt(sqrt(eps(T)))
    Trt2 = sqrt(T(2))
    Trt2i = inv(Trt2)
    c = T[1, 0, 0, 1]
    A = T[0 0 1 0]
    b = T[1]
    G = Diagonal(-one(T) * I, 4)
    h = zeros(T, 4)
    row_idxs = [1, 2, 2]
    col_idxs = [1, 1, 2]
    cones = Cone{T}[Cones.PosSemidefTriSparse{T, Complex{T}}(2, row_idxs, col_idxs)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ Trt2 atol=tol rtol=tol
    @test r.x ≈ [Trt2i, 0, 1, Trt2i] atol=tol rtol=tol
end

function possemideftrisparse3(T; options...)
    if !(T <: LinearAlgebra.BlasReal)
        return # only works with BLAS real types
    end
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    for is_complex in (false, true), side in [1, 2, 5, 20]
        R = (is_complex ? Complex{T} : T)
        rand_mat_L = tril!(sprand(R, side, side, inv(sqrt(side))) + I)
        (row_idxs, col_idxs, vals) = findnz(rand_mat_L)
        dim = (is_complex ? side + 2 * (length(row_idxs) - side) : length(row_idxs))
        c = zeros(T, dim)
        A = zeros(T, 1, dim)
        idx = 1
        for (i, v) in enumerate(vals) # scale
            if row_idxs[i] == col_idxs[i]
                c[idx] = -real(v)
                A[idx] = 1
            else
                c[idx] = -rt2 * real(v)
                if is_complex
                    idx += 1
                    c[idx] = -rt2 * imag(v)
                end
            end
            idx += 1
        end
        b = T[1]
        G = Diagonal(-one(T) * I, dim)
        h = zeros(T, dim)
        cones = Cone{T}[Cones.PosSemidefTriSparse{T, R}(side, row_idxs, col_idxs, use_dual = true)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        eig_max = maximum(eigvals(Hermitian(Matrix(rand_mat_L), :L)))
        @test r.primal_obj ≈ -eig_max atol=tol rtol=tol
    end
end

function possemideftrisparse4(T; options...)
    if !(T <: LinearAlgebra.BlasReal)
        return # only works with BLAS real types
    end
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    c = T[1]
    A = zeros(T, 0, 1)
    b = T[]
    G = zeros(T, 10, 1)
    G[[1, 2, 3, 6, 10]] .= -1
    h = zeros(T, 10)
    @. h[[4, 5, 7, 8, 9]] = rt2 * [1, 1, 1, -1, 1]
    row_idxs = [1, 2, 3, 4, 4, 4, 5, 5, 5, 5]
    col_idxs = [1, 2, 3, 1, 2, 4, 1, 2, 3, 5]
    cones = Cone{T}[Cones.PosSemidefTriSparse{T, T}(5, row_idxs, col_idxs)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    rt3 = sqrt(T(3))
    @test r.primal_obj ≈ rt3 atol=tol rtol=tol
    @test r.s ≈ [rt3, rt3, rt3, rt2, rt2, rt3, rt2, -rt2, rt2, rt3] atol=tol rtol=tol
    inv6 = inv(T(6))
    rt2inv6 = rt2 / 6
    invrt6 = inv(rt2 * rt3)
    @test r.z ≈ [inv6, inv6, inv6, 0, 0, 0, -invrt6, invrt6, -invrt6, inv(T(2))] atol=tol rtol=tol
end

function possemideftrisparse5(T; options...)
    if !(T <: LinearAlgebra.BlasReal)
        return # only works with BLAS real types
    end
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    inv2 = inv(T(2))
    c = vcat(one(T), zeros(T, 5))
    A = zeros(T, 0, 6)
    b = T[]
    G = vcat(Matrix(-one(T) * I, 6, 6), zeros(T, 6, 6))
    G[1, 2:end] .= inv2
    h = vcat(zeros(T, 6), rt2 * [1, 1, 0, 1, -1, 1])
    row_idxs = [1, 2, 3, 4, 5, 4, 4, 4, 5, 5, 5]
    col_idxs = [1, 2, 3, 4, 5, 1, 2, 3, 1, 2, 3]
    cones = Cone{T}[Cones.Nonnegative{T}(1), Cones.PosSemidefTriSparse{T, T}(5, row_idxs, col_idxs, use_dual = true)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    rt3 = sqrt(T(3))
    @test r.primal_obj ≈ rt2 + rt3 atol=tol rtol=tol
    invrt2 = inv(rt2)
    invrt3 = inv(rt3)
    @test r.s ≈ [0, invrt2 + invrt3, invrt2 + invrt3, invrt3, rt2, rt3, rt2, rt2, 0, rt2, -rt2, rt2] atol=tol rtol=tol
    invrt6 = invrt2 * invrt3
    @test r.z ≈ [1, inv2, inv2, inv2, inv2, inv2, -inv2, -inv2, 0, -invrt6, invrt6, -invrt6] atol=tol rtol=tol
end

function hypoperlogdettri1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 4
    for is_complex in (false, true)
        dim = (is_complex ? 2 + side^2 : 2 + Cones.svec_length(side))
        R = (is_complex ? Complex{T} : T)
        c = T[-1, 0]
        A = T[0 1]
        b = T[1]
        G = Matrix{T}(-I, dim, 2)
        mat_half = rand(R, side, side)
        mat = mat_half * mat_half'
        h = zeros(T, dim)
        Cones.smat_to_svec!(view(h, 3:dim), mat, rt2)
        cones = Cone{T}[Cones.HypoPerLogdetTri{T, R}(dim)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.x[1] ≈ -r.primal_obj atol=tol rtol=tol
        @test r.x[2] ≈ 1 atol=tol rtol=tol
        sol_mat = zeros(R, side, side)
        Cones.svec_to_smat!(sol_mat, r.s[3:end] / r.s[2], rt2)
        @test r.s[2] * logdet(cholesky!(Hermitian(sol_mat, :U))) ≈ r.s[1] atol=tol rtol=tol
        Cones.svec_to_smat!(sol_mat, -r.z[3:end] / r.z[1], rt2)
        @test r.z[1] * (logdet(cholesky!(Hermitian(sol_mat, :U))) + T(side)) ≈ r.z[2] atol=tol rtol=tol
    end
end

function hypoperlogdettri2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 3
    for is_complex in (false, true)
        dim = (is_complex ? 2 + side^2 : 2 + Cones.svec_length(side))
        R = (is_complex ? Complex{T} : T)
        c = T[0, 1]
        A = T[1 0]
        b = T[-1]
        G = Matrix{T}(-I, dim, 2)
        mat_half = rand(R, side, side)
        mat = mat_half * mat_half'
        h = zeros(T, dim)
        Cones.smat_to_svec!(view(h, 3:dim), mat, rt2)
        cones = Cone{T}[Cones.HypoPerLogdetTri{T, R}(dim, use_dual = true)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.x[2] ≈ r.primal_obj atol=tol rtol=tol
        @test r.x[1] ≈ -1 atol=tol rtol=tol
        sol_mat = zeros(R, side, side)
        Cones.svec_to_smat!(sol_mat, -r.s[3:end] / r.s[1], rt2)
        @test r.s[1] * (logdet(cholesky!(Hermitian(sol_mat, :U))) + T(side)) ≈ r.s[2] atol=tol rtol=tol
        Cones.svec_to_smat!(sol_mat, r.z[3:end] / r.z[2], rt2)
        @test r.z[2] * logdet(cholesky!(Hermitian(sol_mat, :U))) ≈ r.z[1] atol=tol rtol=tol
    end
end

function hypoperlogdettri3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 3
    for is_complex in (false, true)
        dim = (is_complex ? 2 + side^2 : 2 + Cones.svec_length(side))
        R = (is_complex ? Complex{T} : T)
        c = T[-1, 0]
        A = T[0 1]
        b = T[0]
        G = SparseMatrixCSC(-one(T) * I, dim, 2)
        mat_half = rand(R, side, side)
        mat = mat_half * mat_half'
        h = zeros(T, dim)
        Cones.smat_to_svec!(view(h, 3:dim), mat, rt2)
        cones = Cone{T}[Cones.HypoPerLogdetTri{T, R}(dim)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.x[1] ≈ -r.primal_obj atol=tol rtol=tol
        @test norm(r.x) ≈ 0 atol=tol rtol=tol
    end
end

function hypoperlogdettri4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    c = T[-1, 0, 0, 0, 0]
    A = zeros(T, 1, 5)
    A[1, 2] = 1
    b = T[1]
    G = zeros(T, 7, 5)
    G[1, 1] = G[2, 2] = G[3, 3] = G[5, 5] = -1
    G[4, 4] = -rt2
    G[6, 3] = G[7, 5] = 1
    h = T[0, 0, 0, 0, 0, 1, 1]
    cones = Cone{T}[Cones.HypoPerLogdetTri{T, T}(5), Cones.Nonnegative{T}(2)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=tol rtol=tol
    @test r.x ≈ [0, 1, 1, 0, 1] atol=tol rtol=tol
    @test r.y ≈ [-2] atol=tol rtol=tol
    @test r.z ≈ [-1, -2, 1, 0, 1, 1, 1] atol=tol rtol=tol
end

function hyporootdettri1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 3
    for is_complex in (false, true)
        dim = (is_complex ? 1 + side^2 : 1 + Cones.svec_length(side))
        R = (is_complex ? Complex{T} : T)
        c = T[-1]
        A = zeros(T, 0, 1)
        b = T[]
        G = Matrix{T}(-I, dim, 1)
        mat_half = rand(R, side, side)
        mat = mat_half * mat_half'
        h = zeros(T, dim)
        Cones.smat_to_svec!(view(h, 2:dim), mat, rt2)
        cones = Cone{T}[Cones.HypoRootdetTri{T, R}(dim)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.x[1] ≈ -r.primal_obj atol=tol rtol=tol
        sol_mat = zeros(R, side, side)
        Cones.svec_to_smat!(sol_mat, r.s[2:end], rt2)
        @test det(cholesky!(Hermitian(sol_mat, :U))) ^ inv(T(side)) ≈ r.s[1] atol=tol rtol=tol
        Cones.svec_to_smat!(sol_mat, r.z[2:end] .* T(side), rt2)
        @test det(cholesky!(Hermitian(sol_mat, :U))) ^ inv(T(side)) ≈ -r.z[1] atol=tol rtol=tol
    end
end

function hyporootdettri2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 4
    for is_complex in (false, true)
        dim = (is_complex ? 1 + side^2 : 1 + Cones.svec_length(side))
        R = (is_complex ? Complex{T} : T)
        c = T[1]
        A = zeros(T, 0, 1)
        b = T[]
        G = Matrix{T}(-I, dim, 1)
        mat_half = rand(R, side, side)
        mat = mat_half * mat_half'
        h = zeros(T, dim)
        Cones.smat_to_svec!(view(h, 2:dim), mat, rt2)
        cones = Cone{T}[Cones.HypoRootdetTri{T, R}(dim, use_dual = true)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.x[1] ≈ r.primal_obj atol=tol rtol=tol
        sol_mat = zeros(R, side, side)
        Cones.svec_to_smat!(sol_mat, r.s[2:end] .* T(side), rt2)
        @test det(cholesky!(Hermitian(sol_mat, :U))) ^ inv(T(side)) ≈ -r.s[1] atol=tol rtol=tol
        Cones.svec_to_smat!(sol_mat, r.z[2:end], rt2)
        @test det(cholesky!(Hermitian(sol_mat, :U))) ^ inv(T(side)) ≈ r.z[1] atol=tol rtol=tol
    end
end

function hyporootdettri3(T; options...)
    # max u: u <= rootdet(W) where W is not full rank
    tol = eps(T) ^ 0.15
    rt2 = sqrt(T(2))
    Random.seed!(1)
    side = 3
    for is_complex in (false, true)
        dim = (is_complex ? 1 + side^2 : 1 + Cones.svec_length(side))
        R = (is_complex ? Complex{T} : T)
        c = T[-1]
        A = zeros(T, 0, 1)
        b = T[]
        G = SparseMatrixCSC(-one(T) * I, dim, 1)
        mat_half = T(0.2) * rand(R, side, side - 1)
        mat = mat_half * mat_half'
        h = zeros(T, dim)
        Cones.smat_to_svec!(view(h, 2:dim), mat, rt2)
        cones = Cone{T}[Cones.HypoRootdetTri{T, R}(dim)]

        r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=tol rtol=tol
        @test r.x[1] ≈ zero(T) atol=tol rtol=tol
    end
end

function hyporootdettri4(T; options...)
    tol = sqrt(sqrt(eps(T)))
    rt2 = sqrt(T(2))
    c = T[-1, 0, 0, 0]
    A = zeros(T, 0, 4)
    b = T[]
    G = zeros(T, 6, 4)
    G[1, 1] = G[2, 2] = G[4, 4] = -1
    G[3, 3] = -rt2
    G[5, 2] = G[6, 4] = 1
    h = T[0, 0, 0, 0, 1, 1]
    cones = Cone{T}[Cones.HypoRootdetTri{T, T}(4), Cones.Nonnegative{T}(2)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1 atol=tol rtol=tol
    @test r.x ≈ [1, 1, 0, 1] atol=tol rtol=tol
    @test r.z ≈ [-1, 0.5, 0, 0.5, 0.5, 0.5] atol=tol rtol=tol
end

function wsosinterpnonnegative1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    (U, pts, Ps) = ModelUtilities.interpolate(ModelUtilities.Box{T}(-ones(T, 2), ones(T, 2)), 2)
    DynamicPolynomials.@polyvar x y
    fn = x ^ 4 + x ^ 2 * y ^ 2 + 4 * y ^ 2 + 4

    c = T[-1]
    A = zeros(T, 0, 1)
    b = T[]
    G = ones(T, U, 1)
    h = T[fn(pts[j, :]...) for j in 1:U]
    cones = Cone{T}[Cones.WSOSInterpNonnegative{T, T}(U, Ps)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -T(4) atol=tol rtol=tol
    @test r.x[1] ≈ T(4) atol=tol rtol=tol
end

function wsosinterpnonnegative2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    (U, pts, Ps) = ModelUtilities.interpolate(ModelUtilities.Box{T}(zeros(T, 2), fill(T(3), 2)), 2)
    DynamicPolynomials.@polyvar x y
    fn = (x - 2) ^ 2 + (x * y - 3) ^ 2

    c = T[-1]
    A = zeros(T, 0, 1)
    b = T[]
    G = ones(T, U, 1)
    h = T[fn(pts[j, :]...) for j in 1:U]
    cones = Cone{T}[Cones.WSOSInterpNonnegative{T, T}(U, Ps)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ zero(T) atol=tol rtol=tol
    @test r.x[1] ≈ zero(T) atol=tol rtol=tol
end

function wsosinterpnonnegative3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    (U, pts, Ps) = ModelUtilities.interpolate(ModelUtilities.Box{T}(zeros(T, 2), fill(T(3), 2)), 2)
    DynamicPolynomials.@polyvar x y
    fn = (x - 2) ^ 2 + (x * y - 3) ^ 2

    c = T[fn(pts[j, :]...) for j in 1:U]
    A = ones(T, 1, U)
    b = T[1]
    G = Diagonal(-one(T) * I, U)
    h = zeros(T, U)
    cones = Cone{T}[Cones.WSOSInterpNonnegative{T, T}(U, Ps, use_dual = true)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ zero(T) atol=tol rtol=tol
end

function wsosinterppossemideftri1(T; options...)
    # convexity parameter for (x + 1) ^ 2 * (x - 1) ^ 2
    tol = sqrt(sqrt(eps(T)))
    (U, pts, Ps) = ModelUtilities.interpolate(ModelUtilities.Box{T}([-one(T)], [one(T)]), 1)
    DynamicPolynomials.@polyvar x
    fn = (x + 1) ^ 2 * (x - 1) ^ 2
    H = DynamicPolynomials.differentiate(fn, x, 2)

    c = T[-1]
    A = zeros(T, 0, 1)
    b = T[]
    G = ones(T, U, 1)
    h = T[H(pts[u, :]...) for u in 1:U]
    cones = Cone{T}[Cones.WSOSInterpPosSemidefTri{T}(1, U, Ps)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ T(4) atol=tol rtol=tol
    @test r.x[1] ≈ -T(4) atol=tol rtol=tol
end

function wsosinterppossemideftri2(T; options...)
    # convexity parameter for x[1] ^ 4 - 3 * x[2] ^ 2
    tol = sqrt(sqrt(eps(T)))
    (U, pts, Ps) = ModelUtilities.interpolate(ModelUtilities.FreeDomain{T}(2), 1)
    DynamicPolynomials.@polyvar x[1:2]
    fn = x[1] ^ 4 - 3 * x[2] ^ 2
    H = DynamicPolynomials.differentiate(fn, x, 2)

    c = T[-1]
    A = zeros(T, 0, 1)
    b = T[]
    G = vcat(ones(T, U, 1), zeros(T, U, 1), ones(T, U, 1))
    h = T[H[i, j](pts[u, :]...) for i in 1:2 for j in 1:i for u in 1:U]
    ModelUtilities.vec_to_svec!(h, incr = U)
    cones = Cone{T}[Cones.WSOSInterpPosSemidefTri{T}(2, U, Ps)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ T(6) atol=tol rtol=tol
    @test r.x[1] ≈ -T(6) atol=tol rtol=tol
end

function wsosinterppossemideftri3(T; options...)
    if !(T <: LinearAlgebra.BlasReal)
        return # too slow with BigFloat real types
    end
    # convexity parameter for sum(x .^ 6) - sum(x .^ 2)
    tol = sqrt(sqrt(eps(T)))
    (U, pts, Ps) = ModelUtilities.interpolate(ModelUtilities.FreeDomain{T}(3), 2)
    DynamicPolynomials.@polyvar x[1:3]
    fn = sum(x .^ 4) - sum(x .^ 2)
    H = DynamicPolynomials.differentiate(fn, x, 2)

    c = T[-1]
    A = zeros(T, 0, 1)
    b = T[]
    G = vcat(ones(T, U, 1), zeros(T, U, 1), ones(T, U, 1), zeros(T, U, 1), zeros(T, U, 1), ones(T, U, 1))
    h = T[H[i, j](pts[u, :]...) for i in 1:3 for j in 1:i for u in 1:U]
    ModelUtilities.vec_to_svec!(h, incr = U)
    cones = Cone{T}[Cones.WSOSInterpPosSemidefTri{T}(3, U, Ps)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ T(2) atol=tol rtol=tol
    @test r.x[1] ≈ -T(2) atol=tol rtol=tol
end

function wsosinterpepinormeucl1(T; options...)
    # min t(x) : t(x) ^ 2 >= x ^ 4 on [-1, 1] where t(x) a constant
    tol = sqrt(sqrt(eps(T)))
    (U, pts, Ps) = ModelUtilities.interpolate(ModelUtilities.Box{T}([-one(T)], [one(T)]), 1)
    @assert U == 3
    DynamicPolynomials.@polyvar x
    fn = x ^ 2

    c = ones(T, U)
    A = T[1 -1 0; 1 0 -1]
    b = zeros(T, 2)
    G = vcat(-Matrix{T}(I, U, U), zeros(T, U, U))
    h = vcat(zeros(T, U), T[fn(pts[u, :]...) for u in 1:U])
    cones = Cone{T}[Cones.WSOSInterpEpiNormEucl{T}(2, U, Ps)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ T(U) atol=tol rtol=tol
    @test r.x ≈ ones(T, U) atol=tol rtol=tol
end

function wsosinterpepinormeucl2(T; options...)
    # min t(x) : t(x) ^ 2 >= x ^ 4 + (x - 1) ^ 2 on [-1, 1]^2 where t(x) a constant
    tol = sqrt(sqrt(eps(T)))
    (U, pts, Ps) = ModelUtilities.interpolate(ModelUtilities.Box{T}([-one(T)], [one(T)]), 1)
    DynamicPolynomials.@polyvar x
    fn1 = x ^ 2
    fn2 = (x - 1)

    c = ones(T, U)
    A = T[1 -1 0; 1 0 -1]
    b = zeros(T, 2)
    G = vcat(-Matrix{T}(I, U, U), zeros(T, U, U), zeros(T, U, U))
    h = vcat(zeros(T, U), T[fn1(pts[u, :]...) for u in 1:U], T[fn2(pts[u, :]...) for u in 1:U])
    cones = Cone{T}[Cones.WSOSInterpEpiNormEucl{T}(3, U, Ps)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ sqrt(T(5)) * U atol=tol rtol=tol
    @test r.x ≈ fill(sqrt(T(5)), U) atol=tol rtol=tol
end

function wsosinterpepinormeucl3(T; options...)
    if !(T <: LinearAlgebra.BlasReal)
        return # calc_w only works with BlasReal
    end
    # max: w'f: 25x^4 >= f(x)^4 + 9x^4 on [-1, 1], soln is +/- 4x^2
    tol = sqrt(sqrt(eps(T)))
    (U, pts, Ps, _, w) = ModelUtilities.interpolate(ModelUtilities.Box{T}([-one(T)], [one(T)]), 1, calc_w = true)
    DynamicPolynomials.@polyvar x
    fn1 = 5x^2
    fn2 = 3x^2

    c = -T.(w)
    A = zeros(T, 0, U)
    b = T[]
    G = vcat(spzeros(T, U, U), Diagonal(-one(T) * I, U), spzeros(T, U, U))
    h = vcat(T[fn1(pts[u, :]...) for u in 1:U], zeros(T, U), T[fn2(pts[u, :]...) for u in 1:U])
    cones = Cone{T}[Cones.WSOSInterpEpiNormEucl{T}(3, U, Ps)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -8 / 3 atol=tol rtol=tol
    fn_sol = 4x^2
    @test abs2.(r.x) ≈ abs2.([fn_sol(pts[u, :]...) for u in 1:U]) atol=tol rtol=tol
end

function primalinfeas1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 0]
    A = T[1 1]
    b = [-T(2)]
    G = SparseMatrixCSC(-one(T) * I, 2, 2)
    h = zeros(T, 2)
    cones = Cone{T}[Cones.Nonnegative{T}(2)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :PrimalInfeasible
end

function primalinfeas2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[1, 1, 1]
    A = zeros(T, 0, 3)
    b = T[]
    G = vcat(SparseMatrixCSC(-one(T) * I, 3, 3), Diagonal([one(T), one(T), -one(T)]))
    h = vcat(zeros(T, 3), one(T), one(T), -T(2))
    cones = Cone{T}[Cones.EpiNormEucl{T}(3), Cones.Nonnegative{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :PrimalInfeasible
end

function primalinfeas3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = zeros(T, 3)
    A = SparseMatrixCSC(-one(T) * I, 3, 3)
    b = [one(T), one(T), T(3)]
    G = SparseMatrixCSC(-one(T) * I, 3, 3)
    h = zeros(T, 3)
    cones = Cone{T}[Cones.HypoPerLog{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :PrimalInfeasible
end

function dualinfeas1(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, -1, 0]
    A = zeros(T, 0, 3)
    b = T[]
    G = repeat(SparseMatrixCSC(-one(T) * I, 3, 3), 2, 1)
    h = zeros(T, 6)
    cones = Cone{T}[Cones.EpiNormInf{T, T}(3), Cones.EpiNormInf{T, T}(3, use_dual = true)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :DualInfeasible
end

function dualinfeas2(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[-1, 0]
    A = zeros(T, 0, 2)
    b = T[]
    G = T[-1 0; 0 0; 0 -1]
    h = T[0, 1, 0]
    cones = Cone{T}[Cones.EpiPerSquare{T}(3)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :DualInfeasible
end

function dualinfeas3(T; options...)
    tol = sqrt(sqrt(eps(T)))
    c = T[0, 1, 1, 0]
    A = zeros(T, 0, 4)
    b = T[]
    G = SparseMatrixCSC(-one(T) * I, 4, 4)
    h = zeros(T, 4)
    cones = Cone{T}[Cones.EpiPerSquare{T}(4)]

    r = build_solve_check(c, A, b, G, h, cones; tol = tol, options...)
    @test r.status == :DualInfeasible
end
