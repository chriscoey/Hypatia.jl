#=
Copyright 2018, Chris Coey and contributors
=#

import Random

# solve model, check conic certificates are valid and return certificate data
function solve_and_check(
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    cones::Vector{<:CO.Cone},
    cone_idxs::Vector{UnitRange{Int}},
    system_solver::Type{<:SO.CombinedHSDSystemSolver},
    linear_model::Type{<:MO.LinearModel},
    verbose::Bool;
    atol::Float64 = 1e-4,
    rtol::Float64 = 1e-4,
    )
    model = linear_model(c, A, b, G, h, cones, cone_idxs)
    stepper = SO.CombinedHSDStepper(model, system_solver = system_solver(model))
    solver = SO.HSDSolver(model, verbose = verbose, stepper = stepper)
    SO.solve(solver)
    return SO.get_certificates(solver, model, test = true, atol = atol, rtol = rtol)
end

function dimension1(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[-1, 0]
    A = Matrix{Float64}(undef, 0, 2)
    b = Float64[]
    G = Float64[1 0]
    h = Float64[1]
    cones = [CO.Nonnegative(1, false)]
    cone_idxs = [1:1]

    for use_sparse in (false, true)
        if use_sparse
            A = sparse(A)
            G = sparse(G)
        end
        r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -1 atol=1e-4 rtol=1e-4
        @test r.x ≈ [1, 0] atol=1e-4 rtol=1e-4
        @test isempty(r.y)

        @test_throws ErrorException("some dual equality constraints are inconsistent") linear_model(Float64[-1, -1], A, b, G, h, cones, cone_idxs)
    end
end

function consistent1(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    Random.seed!(1)
    (n, p, q) = (30, 15, 30)
    A = rand(-9.0:9.0, p, n)
    G = Matrix(1.0I, q, n)
    c = rand(0.0:9.0, n)
    rnd1 = rand()
    rnd2 = rand()
    A[11:15, :] = rnd1 * A[1:5, :] - rnd2 * A[6:10, :]
    b = A * ones(n)
    rnd1 = rand()
    rnd2 = rand()
    A[:, 11:15] = rnd1 * A[:, 1:5] - rnd2 * A[:, 6:10]
    G[:, 11:15] = rnd1 * G[:, 1:5] - rnd2 * G[:, 6:10]
    c[11:15] = rnd1 * c[1:5] - rnd2 * c[6:10]
    h = zeros(q)
    cones = [CO.Nonpositive(q)]
    cone_idxs = [1:q]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
end

function inconsistent1(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    Random.seed!(1)
    (n, p, q) = (30, 15, 30)
    A = rand(-9.0:9.0, p, n)
    G = Matrix(-1.0I, q, n)
    c = rand(0.0:9.0, n)
    b = rand(p)
    rnd1 = rand()
    rnd2 = rand()
    A[11:15, :] = rnd1 * A[1:5, :] - rnd2 * A[6:10, :]
    b[11:15] = 2 * (rnd1 * b[1:5] - rnd2 * b[6:10])
    h = zeros(q)

    @test_throws ErrorException("some primal equality constraints are inconsistent") linear_model(c, A, b, G, h, [CO.Nonnegative(q)], [1:q])
end

function inconsistent2(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    Random.seed!(1)
    (n, p, q) = (30, 15, 30)
    A = rand(-9.0:9.0, p, n)
    G = Matrix(-1.0I, q, n)
    c = rand(0.0:9.0, n)
    b = rand(p)
    rnd1 = rand()
    rnd2 = rand()
    A[:,11:15] = rnd1 * A[:,1:5] - rnd2 * A[:,6:10]
    G[:,11:15] = rnd1 * G[:,1:5] - rnd2 * G[:,6:10]
    c[11:15] = 2 * (rnd1 * c[1:5] - rnd2 * c[6:10])
    h = zeros(q)

    @test_throws ErrorException("some dual equality constraints are inconsistent") linear_model(c, A, b, G, h, [CO.Nonnegative(q)], [1:q])
end

function orthant1(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    Random.seed!(1)
    (n, p, q) = (6, 3, 6)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A * ones(n)
    h = zeros(q)
    cone_idxs = [1:q]

    # nonnegative cone
    G = SparseMatrixCSC(-1.0I, q, n)
    cones = [CO.Nonnegative(q)]
    rnn = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test rnn.status == :Optimal

    # nonpositive cone
    G = SparseMatrixCSC(1.0I, q, n)
    cones = [CO.Nonpositive(q)]
    rnp = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test rnp.status == :Optimal

    @test rnp.primal_obj ≈ rnn.primal_obj atol=1e-4 rtol=1e-4
end

function orthant2(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A * ones(n)
    G = rand(q, n) - Matrix(2.0I, q, n)
    h = G * ones(n)
    cone_idxs = [1:q]

    # use dual barrier
    cones = [CO.Nonnegative(q, true)]
    r1 = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r1.status == :Optimal

    # use primal barrier
    cones = [CO.Nonnegative(q, false)]
    r2 = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r2.status == :Optimal

    @test r1.primal_obj ≈ r2.primal_obj atol=1e-4 rtol=1e-4
end

function orthant3(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    Random.seed!(1)
    (n, p, q) = (15, 6, 15)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A * ones(n)
    G = Diagonal(1.0I, n)
    h = zeros(q)
    cone_idxs = [1:q]

    # use dual barrier
    cones = [CO.Nonpositive(q, true)]
    r1 = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r1.status == :Optimal

    # use primal barrier
    cones = [CO.Nonpositive(q, false)]
    r2 = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r2.status == :Optimal

    @test r1.primal_obj ≈ r2.primal_obj atol=1e-4 rtol=1e-4
end

function orthant4(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A * ones(n)
    G = rand(q, n) - Matrix(2.0I, q, n)
    h = G * ones(n)

    # mixture of nonnegative and nonpositive cones
    cones = [CO.Nonnegative(4, false), CO.Nonnegative(6, true)]
    cone_idxs = [1:4, 5:10]
    r1 = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r1.status == :Optimal

    # only nonnegative cone
    cones = [CO.Nonnegative(10, false)]
    cone_idxs = [1:10]
    r2 = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r2.status == :Optimal

    @test r1.primal_obj ≈ r2.primal_obj atol=1e-4 rtol=1e-4
end

function epinorminf1(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[0, -1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, inv(sqrt(2))]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cones = [CO.EpiNormInf(3)]
    cone_idxs = [1:3]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1 - inv(sqrt(2)) atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, inv(sqrt(2)), 1] atol=1e-4 rtol=1e-4
    @test r.y ≈ [1, 1] atol=1e-4 rtol=1e-4
end

function epinorminf2(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    Random.seed!(1)
    c = Float64[1, 0, 0, 0, 0, 0]
    A = rand(-9.0:9.0, 3, 6)
    b = A * ones(6)
    G = rand(6, 6)
    h = G * ones(6)
    cones = [CO.EpiNormInf(6)]
    cone_idxs = [1:6]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 1 atol=1e-4 rtol=1e-4
end

function epinorminf3(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[1, 0, 0, 0, 0, 0]
    A = zeros(0, 6)
    b = zeros(0)
    G = SparseMatrixCSC(-1.0I, 6, 6)
    h = zeros(6)
    cones = [CO.EpiNormInf(6)]
    cone_idxs = [1:6]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=1e-4 rtol=1e-4
    @test r.x ≈ zeros(6) atol=1e-4 rtol=1e-4
end

function epinorminf4(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[0, 1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, -0.4]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cones = [CO.EpiNormInf(3, true)]
    cone_idxs = [1:3]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1 atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, -0.4, 0.6] atol=1e-4 rtol=1e-4
    @test r.y ≈ [1, 0] atol=1e-4 rtol=1e-4
end

function epinorminf5(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    Random.seed!(1)
    c = Float64[1, 0, 0, 0, 0, 0]
    A = rand(-9.0:9.0, 3, 6)
    b = A * ones(6)
    G = rand(6, 6)
    h = G * ones(6)
    cones = [CO.EpiNormInf(6, true)]
    cone_idxs = [1:6]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 1 atol=1e-4 rtol=1e-4
end

function epinorminf6(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    l = 3
    L = 2l + 1
    c = collect(Float64, -l:l)
    A = spzeros(2, L)
    A[1, 1] = A[1, L] = A[2, 1] = 1.0; A[2, L] = -1.0
    b = Float64[0, 0]
    G = [spzeros(1, L); sparse(1.0I, L, L); spzeros(1, L); sparse(2.0I, L, L)]
    h = zeros(2L + 2); h[1] = 1.0; h[L + 2] = 1.0
    cones = [CO.EpiNormInf(L + 1, true), CO.EpiNormInf(L + 1, false)]
    cone_idxs = [1:(L + 1), (L + 2):(2L + 2)]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -l + 1 atol=1e-4 rtol=1e-4
    @test r.x[2] ≈ 0.5 atol=1e-4 rtol=1e-4
    @test r.x[end - 1] ≈ -0.5 atol=1e-4 rtol=1e-4
    @test sum(abs, r.x) ≈ 1 atol=1e-4 rtol=1e-4
end

function epinormeucl1(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[0, -1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, inv(sqrt(2))]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone_idxs = [1:3]

    for is_dual in (true, false)
        cones = [CO.EpiNormEucl(3, is_dual)]

        r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -sqrt(2) atol=1e-4 rtol=1e-4
        @test r.x ≈ [1, inv(sqrt(2)), inv(sqrt(2))] atol=1e-4 rtol=1e-4
        @test r.y ≈ [sqrt(2), 0] atol=1e-4 rtol=1e-4
    end
end

function epinormeucl2(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[0, -1, -1]
    A = Float64[1 0 0]
    b = Float64[0]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone_idxs = [1:3]

    for is_dual in (true, false)
        cones = [CO.EpiNormEucl(3, is_dual)]

        r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x ≈ zeros(3) atol=1e-4 rtol=1e-4
    end
end

function epipersquare1(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[0, 0, -1, -1]
    A = Float64[1 0 0 0; 0 1 0 0]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 4, 4)
    h = zeros(4)
    cone_idxs = [1:4]

    for is_dual in (true, false)
        cones = [CO.EpiPerSquare(4, is_dual)]

        r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -sqrt(2) atol=1e-4 rtol=1e-4
        @test r.x[3:4] ≈ [1, 1] / sqrt(2) atol=1e-4 rtol=1e-4
    end
end

function epipersquare2(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[0, 0, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1/2, 1] / sqrt(2)
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone_idxs = [1:3]

    for is_dual in (true, false)
        cones = [CO.EpiPerSquare(3, is_dual)]

        r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -inv(sqrt(2)) atol=1e-4 rtol=1e-4
        @test r.x[2] ≈ inv(sqrt(2)) atol=1e-4 rtol=1e-4
    end
end

function epipersquare3(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[0, 1, -1, -1]
    A = Float64[1 0 0 0]
    b = Float64[0]
    G = SparseMatrixCSC(-1.0I, 4, 4)
    h = zeros(4)
    cone_idxs = [1:4]

    for is_dual in (true, false)
        cones = [CO.EpiPerSquare(4, is_dual)]

        r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x ≈ zeros(4) atol=1e-4 rtol=1e-4
    end
end

function semidefinite1(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[0, -1, 0]
    A = Float64[1 0 0; 0 0 1]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone_idxs = [1:3]

    for is_dual in (true, false)
        cones = [CO.PosSemidef(3, is_dual)]

        r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -1 atol=1e-4 rtol=1e-4
        @test r.x[2] ≈ 1 atol=1e-4 rtol=1e-4
    end
end

function semidefinite2(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[0, -1, 0]
    A = Float64[1 0 1]
    b = Float64[0]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone_idxs = [1:3]

    for is_dual in (true, false)
        cones = [CO.PosSemidef(3, is_dual)]

        r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x ≈ zeros(3) atol=1e-4 rtol=1e-4
    end
end

function semidefinite3(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[1, 0, 1, 0, 0, 1]
    A = Float64[1 2 3 4 5 6; 1 1 1 1 1 1]
    b = Float64[10, 3]
    G = SparseMatrixCSC(-1.0I, 6, 6)
    h = zeros(6)
    cone_idxs = [1:6]

    for is_dual in (true, false)
        cones = [CO.PosSemidef(6, is_dual)]

        r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 1.249632 atol=1e-4 rtol=1e-4
        @test r.x ≈ [0.491545, 0.647333, 0.426249, 0.571161, 0.531874, 0.331838] atol=1e-4 rtol=1e-4
    end
end

function semidefinitecomplex1(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[1, 0, 0, 1]
    A = Float64[0 0 1 0]
    b = Float64[1]
    G = Diagonal([-1, -sqrt(2), -sqrt(2), -1])
    h = zeros(4)
    cone_idxs = [1:4]
    cones = [CO.PosSemidef{ComplexF64}(4)]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 2 atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, 0, 1, 1] atol=1e-4 rtol=1e-4
end

function hypoperlog1(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[1, 1, 1]
    A = Float64[0 1 0; 1 0 0]
    b = Float64[2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cones = [CO.HypoPerLog()]
    cone_idxs = [1:3]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    ehalf = exp(1 / 2)
    @test r.primal_obj ≈ 2 * ehalf + 3 atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, 2, 2 * ehalf] atol=1e-4 rtol=1e-4
    @test r.y ≈ -[1 + ehalf / 2, 1 + ehalf] atol=1e-4 rtol=1e-4
    @test r.z ≈ c + A' * r.y atol=1e-4 rtol=1e-4
end

function hypoperlog2(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[-1, 0, 0]
    A = Float64[0 1 0]
    b = Float64[0]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cones = [CO.HypoPerLog()]
    cone_idxs = [1:3]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=1e-4 rtol=1e-4
end

function hypoperlog3(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[1, 1, 1]
    A = Matrix{Float64}(undef, 0, 3)
    b = Vector{Float64}(undef, 0)
    G = sparse([1, 2, 3, 4], [1, 2, 3, 1], -ones(4))
    h = zeros(4)
    cones = [CO.HypoPerLog(), CO.Nonnegative(1)]
    cone_idxs = [1:3, 4:4]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=1e-4 rtol=1e-4
    @test r.x ≈ [0, 0, 0] atol=1e-4 rtol=1e-4
    @test isempty(r.y)
end

function hypoperlog4(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[0, 0, 1]
    A = Float64[0 1 0; 1 0 0]
    b = Float64[1, -1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cones = [CO.HypoPerLog(true)]
    cone_idxs = [1:3]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.primal_obj ≈ exp(-2) atol=1e-4 rtol=1e-4
    @test r.x ≈ [-1, 1, exp(-2)] atol=1e-4 rtol=1e-4
end

function epiperpower1(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[1, 0, -1, 0, 0, -1]
    A = Float64[1 1 0 1/2 0 0; 0 0 0 0 1 0]
    b = Float64[2, 1]
    G = SparseMatrixCSC(-1.0I, 6, 6)
    h = zeros(6)
    cones = [CO.EpiPerPower(5.0, false), CO.EpiPerPower(2.5, false)]
    cone_idxs = [1:3, 4:6]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1.80734 atol=1e-4 rtol=1e-4
    @test r.x[[1, 2, 4]] ≈ [0.0639314, 0.783361, 2.30542] atol=1e-4 rtol=1e-4
end

function epiperpower2(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[0, 0, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone_idxs = [1:3]

    for is_dual in (true, false)
        cones = [CO.EpiPerPower(2.0, is_dual)]

        r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? -sqrt(2) : -inv(sqrt(2))) atol=1e-4 rtol=1e-4
        @test r.x[1:2] ≈ [1/2, 1] atol=1e-4 rtol=1e-4
    end
end

function epiperpower3(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[0, 0, 1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[0, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone_idxs = [1:3]

    for is_dual in (true, false), alpha in [1.5; 2.0] # TODO objective gap is large when alpha is different e.g. 2.5, investigate why
        cones = [CO.EpiPerPower(alpha, is_dual)]

        r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose, atol=1e-3, rtol=1e-3)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=1e-3 rtol=1e-3
        @test r.x[1:2] ≈ [0, 1] atol=1e-4 rtol=1e-4
    end
end

function hypogeomean1(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[1, 0, 0, -1, -1, 0]
    A = Float64[1 1 1/2 0 0 0; 0 0 0 0 0 1]
    b = Float64[2, 1]
    G = SparseMatrixCSC(-1.0I, 6, 6)[[4, 1, 2, 5, 3, 6], :]
    h = zeros(6)
    cones = [CO.HypoGeomean([0.2, 0.8], false), CO.HypoGeomean([0.4, 0.6], false)]
    cone_idxs = [1:3, 4:6]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1.80734 atol=1e-4 rtol=1e-4
    @test r.x[1:3] ≈ [0.0639314, 0.783361, 2.30542] atol=1e-4 rtol=1e-4
end

function hypogeomean2(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    c = Float64[-1, 0, 0]
    A = Float64[0 0 1; 0 1 0]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone_idxs = [1:3]

    for is_dual in (true, false)
        cones = [CO.HypoGeomean([0.5, 0.5], is_dual)]

        r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? 0 : -inv(sqrt(2))) atol=1e-4 rtol=1e-4
        @test r.x[2:3] ≈ [1, 0.5] atol=1e-4 rtol=1e-4
    end
end

function hypogeomean3(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    l = 4
    c = vcat(0.0, ones(l))
    A = Float64[1.0 zeros(1, l)]
    G = SparseMatrixCSC(-1.0I, l + 1, l + 1)
    h = zeros(l + 1)
    cone_idxs = [1:(l + 1)]

    for is_dual in (true, false)
        b = is_dual ? [-1.0] : [1.0]
        cones = [CO.HypoGeomean(fill(inv(l), l), is_dual)]

        r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? 1.0 : l) atol=1e-4 rtol=1e-4
        @test r.x[2:end] ≈ (is_dual ? inv(l) : 1.0) * ones(l) atol=1e-4 rtol=1e-4
    end
end

function hypogeomean4(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    l = 4
    c = ones(l)
    A = zeros(0, l)
    b = zeros(0)
    G = Float64[zeros(1, l); Matrix(-1.0I, l, l)]
    h = zeros(l + 1)
    cone_idxs = [1:(l + 1)]

    for is_dual in (true, false)
        cones = [CO.HypoGeomean(fill(inv(l), l), is_dual)]

        r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x ≈ zeros(l) atol=1e-4 rtol=1e-4
    end
end

function epinormspectral1(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    Random.seed!(1)
    (Xn, Xm) = (3, 4)
    Xnm = Xn * Xm
    c = vcat(1.0, zeros(Xnm))
    A = [spzeros(Xnm, 1) sparse(1.0I, Xnm, Xnm)]
    b = rand(Xnm)
    G = sparse(-1.0I, Xnm + 1, Xnm + 1)
    h = vcat(0.0, rand(Xnm))
    cone_idxs = [1:(Xnm + 1)]

    for is_dual in (true, false)
        cones = [CO.EpiNormSpectral(Xn, Xm, is_dual)]

        r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
        @test r.status == :Optimal
        if is_dual
            @test sum(svdvals!(reshape(r.s[2:end], Xn, Xm))) ≈ r.s[1] atol=1e-4 rtol=1e-4
            @test svdvals!(reshape(r.z[2:end], Xn, Xm))[1] ≈ r.z[1] atol=1e-4 rtol=1e-4
        else
            @test svdvals!(reshape(r.s[2:end], Xn, Xm))[1] ≈ r.s[1] atol=1e-4 rtol=1e-4
            @test sum(svdvals!(reshape(r.z[2:end], Xn, Xm))) ≈ r.z[1] atol=1e-4 rtol=1e-4
        end
    end
end

function hypoperlogdet1(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    Random.seed!(1)
    side = 4
    dim = 2 + div(side * (side + 1), 2)
    c = Float64[-1, 0]
    A = Float64[0 1]
    b = Float64[1]
    G = SparseMatrixCSC(-1.0I, dim, 2)
    mat_half = randn(side, side)
    mat = mat_half * mat_half'
    h = zeros(dim)
    CO.smat_to_svec!(view(h, 3:dim), mat)
    cones = [CO.HypoPerLogdet(dim)]
    cone_idxs = [1:dim]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.x[1] ≈ -r.primal_obj atol=1e-4 rtol=1e-4
    @test r.x[2] ≈ 1 atol=1e-4 rtol=1e-4
    @test r.s[2] * logdet(CO.svec_to_smat!(zeros(side, side), r.s[3:end]) / r.s[2]) ≈ r.s[1] atol=1e-4 rtol=1e-4
    @test r.z[1] * (logdet(CO.svec_to_smat!(zeros(side, side), -r.z[3:end]) / r.z[1]) + side) ≈ r.z[2] atol=1e-4 rtol=1e-4
end

function hypoperlogdet2(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    Random.seed!(1)
    side = 3
    dim = 2 + div(side * (side + 1), 2)
    c = Float64[0, 1]
    A = Float64[1 0]
    b = Float64[-1]
    G = SparseMatrixCSC(-1.0I, dim, 2)
    mat_half = rand(side, side)
    mat = mat_half * mat_half'
    h = zeros(dim)
    CO.smat_to_svec!(view(h, 3:dim), mat)
    cones = [CO.HypoPerLogdet(dim, true)]
    cone_idxs = [1:dim]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.x[2] ≈ r.primal_obj atol=1e-4 rtol=1e-4
    @test r.x[1] ≈ -1 atol=1e-4 rtol=1e-4
    @test r.s[1] * (logdet(CO.svec_to_smat!(zeros(side, side), -r.s[3:end]) / r.s[1]) + side) ≈ r.s[2] atol=1e-4 rtol=1e-4
    @test r.z[2] * logdet(CO.svec_to_smat!(zeros(side, side), r.z[3:end]) / r.z[2]) ≈ r.z[1] atol=1e-4 rtol=1e-4
end

function hypoperlogdet3(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    Random.seed!(1)
    side = 3
    dim = 2 + div(side * (side + 1), 2)
    c = Float64[-1, 0]
    A = Float64[0 1]
    b = Float64[0]
    G = SparseMatrixCSC(-1.0I, dim, 2)
    mat_half = rand(side, side)
    mat = mat_half * mat_half'
    h = zeros(dim)
    CO.smat_to_svec!(view(h, 3:dim), mat)
    cones = [CO.HypoPerLogdet(dim)]
    cone_idxs = [1:dim]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.x[1] ≈ -r.primal_obj atol=1e-4 rtol=1e-4
    @test r.x ≈ [0, 0] atol=1e-4 rtol=1e-4
end

function epipersumexp1(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    l = 5
    c = vcat(0.0, -ones(l))
    A = Float64[1 zeros(1, l)]
    b = Float64[1]
    G = Float64[-1 spzeros(1, l); spzeros(1, l + 1); spzeros(l, 1) sparse(-1.0I, l, l)]
    h = zeros(l + 2)
    cones = [CO.EpiPerSumExp(l + 2)]
    cone_idxs = [1:(l + 2)]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.x[1] ≈ 1 atol=1e-4 rtol=1e-4
    @test r.s[2] ≈ 0 atol=1e-4 rtol=1e-4
    @test r.s[1] ≈ 1 atol=1e-4 rtol=1e-4
end

function epipersumexp2(system_solver::Type{<:SO.CombinedHSDSystemSolver}, linear_model::Type{<:MO.LinearModel}, verbose::Bool)
    l = 5
    c = vcat(0.0, -ones(l))
    A = Float64[1 zeros(1, l)]
    b = Float64[1]
    G = Float64[-1.0 spzeros(1, l); spzeros(1, l + 1); spzeros(l, 1) sparse(-1.0I, l, l)]
    h = zeros(l + 2); h[2] = 1.0
    cones = [CO.EpiPerSumExp(l + 2)]
    cone_idxs = [1:(l + 2)]

    r = solve_and_check(c, A, b, G, h, cones, cone_idxs, system_solver, linear_model, verbose)
    @test r.status == :Optimal
    @test r.x[1] ≈ 1 atol=1e-4 rtol=1e-4
    @test r.s[2] ≈ 1 atol=1e-4 rtol=1e-4
    @test r.s[2] * sum(exp, r.s[3:end] / r.s[2]) ≈ r.s[1] atol=1e-4 rtol=1e-4
end
