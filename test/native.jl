#=
Copyright 2018, Chris Coey and contributors
=#

function dimension1(; verbose, linearsystem)
    A = Matrix{Float64}(undef, 0, 2)
    b = Float64[]
    G = [1.0 0.0]
    h = [1.0]
    c = [-1.0, 0.0]
    cone = CO.Cone([CO.Nonnegative(1, false)], [1:1])

    mdl = HYP.Model(verbose=verbose)
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    r.status == :Optimal
    @test r.pobj ≈ -1 atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, 0] atol=1e-4 rtol=1e-4
    @test isempty(r.y)

    c = [-1.0, -1.0]
    HYP.check_data(c, A, b, G, h, cone)
    @test_throws ErrorException("some dual equality constraints are inconsistent") HYP.preprocess_data(c, A, b, G, useQR=true)

    A = sparse(A)
    G = sparse(G)
    c = [-1.0, 0.0]
    mdl = HYP.Model(verbose=verbose)
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    r.status == :Optimal
    @test r.pobj ≈ -1 atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, 0] atol=1e-4 rtol=1e-4
    @test isempty(r.y)

    c = [-1.0, -1.0]
    HYP.check_data(c, A, b, G, h, cone)
    @test_throws ErrorException("some dual equality constraints are inconsistent") HYP.preprocess_data(c, A, b, G, useQR=true)
end

function consistent1(; verbose, linearsystem)
    mdl = HYP.Model(verbose=verbose)
    Random.seed!(1)
    (n, p, q) = (30, 15, 30)
    A = rand(-9.0:9.0, p, n)
    G = Matrix(1.0I, q, n)
    c = rand(0.0:9.0, n)
    rnd1 = rand()
    rnd2 = rand()
    A[11:15,:] = rnd1*A[1:5,:] - rnd2*A[6:10,:]
    b = A*ones(n)
    rnd1 = rand()
    rnd2 = rand()
    A[:,11:15] = rnd1*A[:,1:5] - rnd2*A[:,6:10]
    G[:,11:15] = rnd1*G[:,1:5] - rnd2*G[:,6:10]
    c[11:15] = rnd1*c[1:5] - rnd2*c[6:10]
    h = zeros(q)
    cone = CO.Cone([CO.Nonpositive(q)], [1:q])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
end

function inconsistent1(; verbose, linearsystem)
    mdl = HYP.Model(verbose=verbose)
    Random.seed!(1)
    (n, p, q) = (30, 15, 30)
    A = rand(-9.0:9.0, p, n)
    G = Matrix(-1.0I, q, n)
    c = rand(0.0:9.0, n)
    b = rand(p)
    rnd1 = rand()
    rnd2 = rand()
    A[11:15,:] = rnd1*A[1:5,:] - rnd2*A[6:10,:]
    b[11:15] = 2*(rnd1*b[1:5] - rnd2*b[6:10])
    h = zeros(q)
    cone = CO.Cone([CO.Nonnegative(q)], [1:q])
    @test_throws ErrorException("some primal equality constraints are inconsistent") solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
end

function inconsistent2(; verbose, linearsystem)
    mdl = HYP.Model(verbose=verbose)
    Random.seed!(1)
    (n, p, q) = (30, 15, 30)
    A = rand(-9.0:9.0, p, n)
    G = Matrix(-1.0I, q, n)
    c = rand(0.0:9.0, n)
    b = rand(p)
    rnd1 = rand()
    rnd2 = rand()
    A[:,11:15] = rnd1*A[:,1:5] - rnd2*A[:,6:10]
    G[:,11:15] = rnd1*G[:,1:5] - rnd2*G[:,6:10]
    c[11:15] = 2*(rnd1*c[1:5] - rnd2*c[6:10])
    h = zeros(q)
    cone = CO.Cone([CO.Nonnegative(q)], [1:q])
    @test_throws ErrorException("some dual equality constraints are inconsistent") solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
end

function orthant1(; verbose, linearsystem)
    Random.seed!(1)
    (n, p, q) = (6, 3, 6)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    h = zeros(q)

    # nonnegative cone
    mdl = HYP.Model(verbose=verbose)
    G = SparseMatrixCSC(-1.0I, q, n)
    cone = CO.Cone([CO.Nonnegative(q)], [1:q])
    rnn = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test rnn.status == :Optimal

    # nonpositive cone
    mdl = HYP.Model(verbose=verbose)
    G = SparseMatrixCSC(1.0I, q, n)
    cone = CO.Cone([CO.Nonpositive(q)], [1:q])
    rnp = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test rnp.status == :Optimal

    @test rnp.pobj ≈ rnn.pobj atol=1e-4 rtol=1e-4
end

function orthant2(; verbose, linearsystem)
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    G = rand(q, n) - Matrix(2.0I, q, n)
    h = G*ones(n)

    mdl = HYP.Model(verbose=verbose)
    cone = CO.Cone([CO.Nonnegative(q, true)], [1:q])
    r1 = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r1.status == :Optimal

    mdl = HYP.Model(verbose=verbose)
    cone = CO.Cone([CO.Nonnegative(q, false)], [1:q])
    r2 = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r2.status == :Optimal

    @test r1.pobj ≈ r2.pobj atol=1e-4 rtol=1e-4
end

function orthant3(; verbose, linearsystem)
    Random.seed!(1)
    (n, p, q) = (15, 6, 15)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    G = Diagonal(1.0I, n)
    h = zeros(q)

    mdl = HYP.Model(verbose=verbose)
    cone = CO.Cone([CO.Nonpositive(q, true)], [1:q])
    r1 = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r1.status == :Optimal

    mdl = HYP.Model(verbose=verbose)
    cone = CO.Cone([CO.Nonpositive(q, false)], [1:q])
    r2 = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r2.status == :Optimal

    @test r1.pobj ≈ r2.pobj atol=1e-4 rtol=1e-4
end

function orthant4(; verbose, linearsystem)
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    G = rand(q, n) - Matrix(2.0I, q, n)
    h = G*ones(n)

    mdl = HYP.Model(verbose=verbose)
    cone = CO.Cone([CO.Nonnegative(4, false), CO.Nonnegative(6, true)], [1:4, 5:10])
    r1 = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r1.status == :Optimal

    mdl = HYP.Model(verbose=verbose)
    cone = CO.Cone([CO.Nonnegative(10, false)], [1:10])
    r2 = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r2.status == :Optimal

    @test r1.pobj ≈ r2.pobj atol=1e-4 rtol=1e-4
end

function epinorminf1(; verbose, linearsystem)
    mdl = HYP.Model(verbose=verbose)
    c = Float64[0, -1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, 1/sqrt(2)]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = CO.Cone([CO.EpiNormInf(3)], [1:3])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ -1 - 1/sqrt(2) atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, 1/sqrt(2), 1] atol=1e-4 rtol=1e-4
    @test r.y ≈ [1, 1] atol=1e-4 rtol=1e-4
end

function epinorminf2(; verbose, linearsystem)
    mdl = HYP.Model(verbose=verbose)
    Random.seed!(1)
    c = Float64[1, 0, 0, 0, 0, 0]
    A = rand(-9.0:9.0, 3, 6)
    b = A*ones(6)
    G = rand(6, 6)
    h = G*ones(6)
    cone = CO.Cone([CO.EpiNormInf(6)], [1:6])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ 1 atol=1e-4 rtol=1e-4
end

function epinorminf3(; verbose, linearsystem)
    mdl = HYP.Model(verbose=verbose)
    Random.seed!(1)
    c = Float64[1, 0, 0, 0, 0, 0]
    A = zeros(0, 6)
    b = zeros(0)
    G = SparseMatrixCSC(-1.0I, 6, 6)
    h = zeros(6)
    cone = CO.Cone([CO.EpiNormInf(6)], [1:6])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
    @test r.x ≈ zeros(6) atol=1e-4 rtol=1e-4
end

function epinorminf4(; verbose, linearsystem)
    mdl = HYP.Model(verbose=verbose)
    c = Float64[0, 1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, -0.4]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = CO.Cone([CO.EpiNormInf(3, true)], [1:3])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ -1 atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, -0.4, 0.6] atol=1e-4 rtol=1e-4
    @test r.y ≈ [1, 0] atol=1e-4 rtol=1e-4
end

function epinorminf5(; verbose, linearsystem)
    mdl = HYP.Model(verbose=verbose)
    Random.seed!(1)
    c = Float64[1, 0, 0, 0, 0, 0]
    A = rand(-9.0:9.0, 3, 6)
    b = A*ones(6)
    G = rand(6, 6)
    h = G*ones(6)
    cone = CO.Cone([CO.EpiNormInf(6, true)], [1:6])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 15
    @test r.pobj ≈ 1 atol=1e-4 rtol=1e-4
end

function epinorminf6(; verbose, linearsystem)
    Random.seed!(1)
    l = 3
    L = 2l + 1
    c = collect(-Float64(l):Float64(l))
    A = spzeros(2, L)
    A[1,1] = A[1,L] = A[2,1] = 1.0; A[2,L] = -1.0
    b = [0.0, 0.0]
    G = [spzeros(1, L); sparse(1.0I, L, L); spzeros(1, L); sparse(2.0I, L, L)]
    h = zeros(2L+2); h[1] = 1.0; h[L+2] = 1.0
    mdl = HYP.Model(verbose=verbose)
    cone = CO.Cone([CO.EpiNormInf(L+1, true), CO.EpiNormInf(L+1, false)], [1:L+1, L+2:2L+2])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 25
    @test r.pobj ≈ -l + 1 atol=1e-4 rtol=1e-4
    @test r.x[2] ≈ 0.5 atol=1e-4 rtol=1e-4
    @test r.x[end-1] ≈ -0.5 atol=1e-4 rtol=1e-4
    @test sum(abs, r.x) ≈ 1.0 atol=1e-4 rtol=1e-4
end

function epinormeucl1(; verbose, linearsystem)
    c = Float64[0, -1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, 1/sqrt(2)]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for isdual in [true, false]
        mdl = HYP.Model(verbose=verbose)
        cone = CO.Cone([CO.EpiNormEucl(3, isdual)], [1:3])
        r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ -sqrt(2) atol=1e-4 rtol=1e-4
        @test r.x ≈ [1, 1/sqrt(2), 1/sqrt(2)] atol=1e-4 rtol=1e-4
        @test r.y ≈ [sqrt(2), 0] atol=1e-4 rtol=1e-4
    end
end

function epinormeucl2(; verbose, linearsystem)
    c = Float64[0, -1, -1]
    A = Float64[1 0 0]
    b = Float64[0]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for isdual in [true, false]
        mdl = HYP.Model(verbose=verbose)
        cone = CO.Cone([CO.EpiNormEucl(3, isdual)], [1:3])
        r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x ≈ zeros(3) atol=1e-4 rtol=1e-4
    end
end

function epipersquare1(; verbose, linearsystem)
    c = Float64[0, 0, -1, -1]
    A = Float64[1 0 0 0; 0 1 0 0]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 4, 4)
    h = zeros(4)

    for isdual in [true, false]
        mdl = HYP.Model(verbose=verbose)
        cone = CO.Cone([CO.EpiPerSquare(4, isdual)], [1:4])
        r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ -sqrt(2) atol=1e-4 rtol=1e-4
        @test r.x[3:4] ≈ [1, 1]/sqrt(2) atol=1e-4 rtol=1e-4
    end
end

function epipersquare2(; verbose, linearsystem)
    c = Float64[0, 0, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1/2, 1]/sqrt(2)
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for isdual in [true, false]
        mdl = HYP.Model(verbose=verbose)
        cone = CO.Cone([CO.EpiPerSquare(3, isdual)], [1:3])
        r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
        @test r.status == :Optimal
        @test r.niters <= 15
        @test r.pobj ≈ -1/sqrt(2) atol=1e-4 rtol=1e-4
        @test r.x[2] ≈ 1/sqrt(2) atol=1e-4 rtol=1e-4
    end
end

function epipersquare3(; verbose, linearsystem)
    c = Float64[0, 1, -1, -1]
    A = Float64[1 0 0 0]
    b = Float64[0]
    G = SparseMatrixCSC(-1.0I, 4, 4)
    h = zeros(4)

    for isdual in [true, false]
        mdl = HYP.Model(verbose=verbose)
        cone = CO.Cone([CO.EpiPerSquare(4, isdual)], [1:4])
        r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x ≈ zeros(4) atol=1e-4 rtol=1e-4
    end
end

function semidefinite1(; verbose, linearsystem)
    c = Float64[0, -1, 0]
    A = Float64[1 0 0; 0 0 1]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for isdual in [true, false]
        mdl = HYP.Model(verbose=verbose)
        cone = CO.Cone([CO.PosSemidef(3, isdual)], [1:3])
        r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ -1 atol=1e-4 rtol=1e-4
        @test r.x[2] ≈ 1 atol=1e-4 rtol=1e-4
    end
end

function semidefinite2(; verbose, linearsystem)
    c = Float64[0, -1, 0]
    A = Float64[1 0 1]
    b = Float64[0]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for isdual in [true, false]
        mdl = HYP.Model(verbose=verbose)
        cone = CO.Cone([CO.PosSemidef(3, isdual)], [1:3])
        r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x ≈ zeros(3) atol=1e-4 rtol=1e-4
    end
end

function semidefinite3(; verbose, linearsystem)
    c = Float64[1, 0, 1, 0, 0, 1]
    A = Float64[1 2 3 4 5 6; 1 1 1 1 1 1]
    b = Float64[10, 3]
    G = SparseMatrixCSC(-1.0I, 6, 6)
    h = zeros(6)

    for isdual in [true, false]
        mdl = HYP.Model(verbose=verbose)
        cone = CO.Cone([CO.PosSemidef(6, isdual)], [1:6])
        r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ 1.249632 atol=1e-4 rtol=1e-4
        @test r.x ≈ [0.491545, 0.647333, 0.426249, 0.571161, 0.531874, 0.331838] atol=1e-4 rtol=1e-4
    end
end

function hypoperlog1(; verbose, linearsystem)
    mdl = HYP.Model(verbose=verbose)
    c = Float64[1, 1, 1]
    A = Float64[0 1 0; 1 0 0]
    b = Float64[2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = CO.Cone([CO.HypoPerLog()], [1:3])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ 2*exp(1/2)+3 atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, 2, 2*exp(1/2)] atol=1e-4 rtol=1e-4
    @test r.y ≈ -[1+exp(1/2)/2, 1+exp(1/2)] atol=1e-4 rtol=1e-4
    @test r.z ≈ c+A'*r.y atol=1e-4 rtol=1e-4
end

function hypoperlog2(; verbose, linearsystem)
    mdl = HYP.Model(verbose=verbose)
    c = Float64[-1, 0, 0]
    A = Float64[0 1 0]
    b = Float64[0]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = CO.Cone([CO.HypoPerLog()], [1:3])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 25
    @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
end

function hypoperlog3(; verbose, linearsystem)
    mdl = HYP.Model(verbose=verbose)
    c = Float64[1, 1, 1]
    A = Matrix{Float64}(undef, 0, 3)
    b = Vector{Float64}(undef, 0)
    G = sparse([1, 2, 3, 4], [1, 2, 3, 1], -ones(4))
    h = zeros(4)
    cone = CO.Cone([CO.HypoPerLog(), CO.Nonnegative(1)], [1:3, 4:4])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
    @test r.x ≈ [0, 0, 0] atol=1e-4 rtol=1e-4
    @test isempty(r.y)
end

function hypoperlog4(; verbose, linearsystem)
    mdl = HYP.Model(verbose=verbose)
    c = Float64[0, 0, 1]
    A = Float64[0 1 0; 1 0 0]
    b = Float64[1, -1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = CO.Cone([CO.HypoPerLog(true)], [1:3])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ exp(-2) atol=1e-4 rtol=1e-4
    @test r.x ≈ [-1, 1, exp(-2)] atol=1e-4 rtol=1e-4
end

function epiperpower1(; verbose, linearsystem)
    mdl = HYP.Model(verbose=verbose)
    c = Float64[1, 0, -1, 0, 0, -1]
    A = Float64[1 1 0 1/2 0 0; 0 0 0 0 1 0]
    b = Float64[2, 1]
    G = SparseMatrixCSC(-1.0I, 6, 6)
    h = zeros(6)
    cone = CO.Cone([CO.EpiPerPower(5.0, false), CO.EpiPerPower(2.5, false)], [1:3, 4:6])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ -1.80734 atol=1e-4 rtol=1e-4
    @test r.x[[1,2,4]] ≈ [0.0639314, 0.783361, 2.30542] atol=1e-4 rtol=1e-4
end

function epiperpower2(; verbose, linearsystem)
    c = Float64[0, 0, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for isdual in [true, false]
        mdl = HYP.Model(verbose=verbose)
        cone = CO.Cone([CO.EpiPerPower(2.0, isdual)], [1:3])
        r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ (isdual ? -sqrt(2) : -1/sqrt(2)) atol=1e-4 rtol=1e-4
        @test r.x[1:2] ≈ [1/2, 1] atol=1e-4 rtol=1e-4
    end
end

function epiperpower3(; verbose, linearsystem)
    c = Float64[0, 0, 1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[0, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for isdual in [true, false]
        mdl = HYP.Model(verbose=verbose, tolfeas=1e-9)
        cone = CO.Cone([CO.EpiPerPower(2.0, isdual)], [1:3])
        r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
        @test r.status == :Optimal
        @test r.niters <= 50
        @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x[1:2] ≈ [0, 1] atol=1e-4 rtol=1e-4
    end
end

function hypogeomean1(; verbose, linearsystem)
    mdl = HYP.Model(verbose=verbose)
    c = Float64[1, 0, 0, -1, -1, 0]
    A = Float64[1 1 1/2 0 0 0; 0 0 0 0 0 1]
    b = Float64[2, 1]
    G = SparseMatrixCSC(-1.0I, 6, 6)[[4, 1, 2, 5, 3, 6], :]
    h = zeros(6)
    cone = CO.Cone([CO.HypoGeomean([0.2, 0.8], false), CO.HypoGeomean([0.4, 0.6], false)], [1:3, 4:6])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 25
    @test r.pobj ≈ -1.80734 atol=1e-4 rtol=1e-4
    @test r.x[1:3] ≈ [0.0639314, 0.783361, 2.30542] atol=1e-4 rtol=1e-4
end

function hypogeomean2(; verbose, linearsystem)
    c = Float64[-1, 0, 0]
    A = Float64[0 0 1; 0 1 0]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for isdual in [true, false]
        mdl = HYP.Model(verbose=verbose)
        cone = CO.Cone([CO.HypoGeomean([0.5, 0.5], isdual)], [1:3])
        r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ (isdual ? 0 : -1/sqrt(2)) atol=1e-4 rtol=1e-4
        @test r.x[2:3] ≈ [1, 0.5] atol=1e-4 rtol=1e-4
    end
end

function hypogeomean3(; verbose, linearsystem)
    l = 4
    c = vcat(0.0, ones(l))
    A = [1.0 zeros(1, l)]
    G = SparseMatrixCSC(-1.0I, l+1, l+1)
    h = zeros(l+1)

    for isdual in [true, false]
        b = (isdual ? [-1.0] : [1.0])
        mdl = HYP.Model(verbose=verbose)
        cone = CO.Cone([CO.HypoGeomean(fill(1/l, l), isdual)], [1:l+1])
        r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
        @test r.status == :Optimal
        @test r.niters <= 25
        @test r.pobj ≈ (isdual ? 1 : l) atol=1e-4 rtol=1e-4
        @test r.x[2:end] ≈ (isdual ? fill(1/l, l) : ones(l)) atol=1e-4 rtol=1e-4
    end
end

function hypogeomean4(; verbose, linearsystem)
    l = 4
    c = ones(l)
    A = zeros(0, l)
    b = zeros(0)
    G = [zeros(1, l); Matrix(-1.0I, l, l)]
    h = zeros(l+1)

    for isdual in [true, false]
        mdl = HYP.Model(verbose=verbose)
        cone = CO.Cone([CO.HypoGeomean(fill(1/l, l), isdual)], [1:l+1])
        r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
        @test r.status == :Optimal
        @test r.niters <= 15
        @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x ≈ zeros(l) atol=1e-4 rtol=1e-4
    end
end

function epinormspectral1(; verbose, linearsystem)
    Random.seed!(1)
    (Xn, Xm) = (3, 4)
    Xnm = Xn*Xm
    c = vcat(1.0, zeros(Xnm))
    p = 0
    A = [spzeros(Xnm, 1) sparse(1.0I, Xnm, Xnm)]
    b = rand(Xnm)
    G = sparse(-1.0I, Xnm+1, Xnm+1)
    h = vcat(0.0, rand(Xnm))

    for isdual in [true, false]
        mdl = HYP.Model(verbose=verbose)
        cone = CO.Cone([CO.EpiNormSpectral(Xn, Xm, isdual)], [1:Xnm+1])
        r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
        @test r.status == :Optimal
        @test r.niters <= 20
        if isdual
            @test sum(svdvals!(reshape(r.s[2:end], Xn, Xm))) ≈ r.s[1] atol=1e-4 rtol=1e-4
            @test svdvals!(reshape(r.z[2:end], Xn, Xm))[1] ≈ r.z[1] atol=1e-4 rtol=1e-4
        else
            @test svdvals!(reshape(r.s[2:end], Xn, Xm))[1] ≈ r.s[1] atol=1e-4 rtol=1e-4
            @test sum(svdvals!(reshape(r.z[2:end], Xn, Xm))) ≈ r.z[1] atol=1e-4 rtol=1e-4
        end
    end
end

function hypoperlogdet1(; verbose, linearsystem)
    Random.seed!(1)
    side = 4
    dim = round(Int, 2 + side*(side + 1)/2)
    c = [-1.0, 0.0]
    A = [0.0 1.0]
    b = [1.0]
    G = SparseMatrixCSC(-1.0I, dim, 2)
    mathalf = rand(side, side)
    mat = mathalf*mathalf'
    h = zeros(dim)
    CO.mattovec!(view(h, 3:dim), mat)
    mdl = HYP.Model(verbose=verbose)
    cone = CO.Cone([CO.HypoPerLogdet(dim)], [1:dim])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 30
    @test r.x[1] ≈ -r.pobj atol=1e-4 rtol=1e-4
    @test r.x[2] ≈ 1 atol=1e-4 rtol=1e-4
    @test r.s[2]*logdet(CO.vectomat!(zeros(side, side), r.s[3:end])/r.s[2]) ≈ r.s[1] atol=1e-4 rtol=1e-4
    @test r.z[1]*(logdet(CO.vectomat!(zeros(side, side), -r.z[3:end])/r.z[1]) + side) ≈ r.z[2] atol=1e-4 rtol=1e-4
end

function hypoperlogdet2(; verbose, linearsystem)
    Random.seed!(1)
    side = 3
    dim = round(Int, 2 + side*(side + 1)/2)
    c = [0.0, 1.0]
    A = [1.0 0.0]
    b = [-1.0]
    G = SparseMatrixCSC(-1.0I, dim, 2)
    mathalf = rand(side, side)
    mat = mathalf*mathalf'
    h = zeros(dim)
    CO.mattovec!(view(h, 3:dim), mat)
    mdl = HYP.Model(verbose=verbose)
    cone = CO.Cone([CO.HypoPerLogdet(dim, true)], [1:dim])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 25
    @test r.x[2] ≈ r.pobj atol=1e-4 rtol=1e-4
    @test r.x[1] ≈ -1 atol=1e-4 rtol=1e-4
    @test r.s[1]*(logdet(CO.vectomat!(zeros(side, side), -r.s[3:end])/r.s[1]) + side) ≈ r.s[2] atol=1e-4 rtol=1e-4
    @test r.z[2]*logdet(CO.vectomat!(zeros(side, side), r.z[3:end])/r.z[2]) ≈ r.z[1] atol=1e-4 rtol=1e-4
end

function hypoperlogdet3(; verbose, linearsystem)
    Random.seed!(1)
    side = 3
    dim = round(Int, 2 + side*(side + 1)/2)
    c = [-1.0, 0.0]
    A = [0.0 1.0]
    b = [0.0]
    G = SparseMatrixCSC(-1.0I, dim, 2)
    mathalf = rand(side, side)
    mat = mathalf*mathalf'
    h = zeros(dim)
    CO.mattovec!(view(h, 3:dim), mat)
    mdl = HYP.Model(verbose=verbose)
    cone = CO.Cone([CO.HypoPerLogdet(dim)], [1:dim])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 30
    @test r.x[1] ≈ -r.pobj atol=1e-4 rtol=1e-4
    @test r.x ≈ [0, 0] atol=1e-4 rtol=1e-4
end

function epipersumexp1(; verbose, linearsystem)
    l = 5
    c = vcat(0.0, -ones(l))
    A = [1.0 zeros(1, l)]
    b = [1.0]
    G = [-1.0 spzeros(1, l); spzeros(1, l+1); spzeros(l, 1) sparse(-1.0I, l, l)]
    h = zeros(l+2)
    mdl = HYP.Model(verbose=verbose)
    cone = CO.Cone([CO.EpiPerSumExp(l+2)], [1:l+2])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 30
    @test r.x[1] ≈ 1 atol=1e-4 rtol=1e-4
    @test r.s[2] ≈ 0 atol=1e-4 rtol=1e-4
    @test r.s[1] ≈ 1 atol=1e-4 rtol=1e-4
end

function epipersumexp2(; verbose, linearsystem)
    l = 5
    c = vcat(0.0, -ones(l))
    A = [1.0 zeros(1, l)]
    b = [1.0]
    G = [-1.0 spzeros(1, l); spzeros(1, l+1); spzeros(l, 1) sparse(-1.0I, l, l)]
    h = zeros(l+2); h[2] = 1.0
    mdl = HYP.Model(verbose=verbose)
    cone = CO.Cone([CO.EpiPerSumExp(l+2)], [1:l+2])
    r = solveandcheck(mdl, c, A, b, G, h, cone, linearsystem)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.x[1] ≈ 1 atol=1e-4 rtol=1e-4
    @test r.s[2] ≈ 1 atol=1e-4 rtol=1e-4
    @test r.s[2]*sum(exp, r.s[3:end]/r.s[2]) ≈ r.s[1] atol=1e-4 rtol=1e-4
end
