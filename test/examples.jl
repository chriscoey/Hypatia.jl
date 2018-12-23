#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

function _envelope1(; verbose, lscachetype)
    # dense methods
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_envelope(2, 5, 1, 5, use_data=true, usedense=true)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.pobj ≈ 25.502777 atol=1e-4 rtol=1e-4
    @test r.niters <= 35

    # sparse methods
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_envelope(2, 5, 1, 5, use_data=true, usedense=false)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.pobj ≈ 25.502777 atol=1e-4 rtol=1e-4
    @test r.niters <= 35
end

function _envelope2(; verbose, lscachetype)
    # dense methods
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_envelope(2, 4, 2, 7, usedense=true)
    rd = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test rd.status == :Optimal
    @test rd.niters <= 60

    # sparse methods
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_envelope(2, 4, 2, 7, usedense=false)
    rs = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test rs.status == :Optimal
    @test rs.niters <= 60

    @test rs.pobj ≈ rd.pobj atol=1e-4 rtol=1e-4
end

function _envelope3(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_envelope(2, 3, 3, 5, usedense=false)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 60
end

function _envelope4(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_envelope(2, 2, 4, 3, usedense=false)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 55
end

function _linearopt1(; verbose, lscachetype)
    # dense methods
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_linearopt(25, 50, usedense=true, tosparse=false)
    rd = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test rd.status == :Optimal
    @test rd.niters <= 35

    # sparse methods
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_linearopt(25, 50, usedense=true, tosparse=true)
    rs = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test rs.status == :Optimal
    @test rs.niters <= 35

    @test rs.pobj ≈ rd.pobj atol=1e-4 rtol=1e-4
end

function _linearopt2(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose, tolfeas=1e-9)
    (c, A, b, G, h, cone) = build_linearopt(500, 1000, use_data=true, usedense=true)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 70
    @test r.pobj ≈ 2055.807 atol=1e-4 rtol=1e-4
end

# for namedpoly tests, most optimal values are taken from https://people.sc.fsu.edu/~jburkardt/py_src/polynomials/polynomials.html

function _namedpoly1(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_namedpoly(:butcher, 2)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 45
    @test r.pobj ≈ -1.4393333333 atol=1e-4 rtol=1e-4
end

function _namedpoly2(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_namedpoly(:caprasse, 4)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 45
    @test r.pobj ≈ -3.1800966258 atol=1e-4 rtol=1e-4
end

function _namedpoly3(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose, tolfeas=1e-9)
    (c, A, b, G, h, cone) = build_namedpoly(:goldsteinprice, 6)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype, atol=2e-3)
    @test r.status == :Optimal
    @test r.niters <= 70
    @test r.pobj ≈ 3 atol=1e-4 rtol=1e-4
end

function _namedpoly4(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_namedpoly(:heart, 2)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    # @test r.niters <= 40
    @test r.pobj ≈ -1.36775 atol=1e-4 rtol=1e-4
end

function _namedpoly5(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_namedpoly(:lotkavolterra, 3)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 40
    @test r.pobj ≈ -20.8 atol=1e-4 rtol=1e-4
end

function _namedpoly6(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_namedpoly(:magnetism7, 2)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 35
    @test r.pobj ≈ -0.25 atol=1e-4 rtol=1e-4
end

function _namedpoly7(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_namedpoly(:motzkin, 7)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 45
    @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
end

function _namedpoly8(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_namedpoly(:reactiondiffusion, 4)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 40
    @test r.pobj ≈ -36.71269068 atol=1e-4 rtol=1e-4
end

function _namedpoly9(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_namedpoly(:robinson, 8)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 40
    @test r.pobj ≈ 0.814814 atol=1e-4 rtol=1e-4
end

function _namedpoly10(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose, tolfeas=2e-10)
    (c, A, b, G, h, cone) = build_namedpoly(:rosenbrock, 5)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype, atol=1e-3)
    @test r.status == :Optimal
    @test r.niters <= 70
    @test r.pobj ≈ 0 atol=1e-3 rtol=1e-3
end

function _namedpoly11(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose, tolfeas=1e-10)
    (c, A, b, G, h, cone) = build_namedpoly(:schwefel, 4)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype, atol=1e-3)
    @test r.status == :Optimal
    @test r.niters <= 65
    @test r.pobj ≈ 0 atol=1e-3 rtol=1e-3
end

function solveandcheck_JuMP(mdl, truemin)
    JuMP.optimize!(mdl)
    term_status = JuMP.termination_status(mdl)
    pobj = JuMP.objective_value(mdl)
    dobj = JuMP.objective_bound(mdl)
    pr_status = JuMP.primal_status(mdl)
    du_status = JuMP.dual_status(mdl)
    @test term_status == MOI.OPTIMAL
    @test pr_status == MOI.FEASIBLE_POINT
    @test du_status == MOI.FEASIBLE_POINT
    @test pobj ≈ dobj atol=1e-3 rtol=1e-3
    @test pobj ≈ truemin atol=1e-3 rtol=1e-3
end

function _namedpoly1_JuMP()
    # the Heart polynomial in a box
    (x, f, dom, truemin) = getpolydata(:heart)
    # WSOS formulation
    mdl = build_JuMP_namedpoly_WSOS(x, f, dom, d=2)
    solveandcheck_JuMP(mdl, truemin)
end

function _namedpoly2_JuMP()
    # the Schwefel polynomial in a box
    (x, f, dom, truemin) = getpolydata(:schwefel)
    # WSOS formulation
    mdl = build_JuMP_namedpoly_WSOS(x, f, dom, d=2)
    solveandcheck_JuMP(mdl, truemin)
    # SDP formulation
    # mdl = build_JuMP_namedpoly_PSD(x, f, dom, d=2)
    # solveandcheck_JuMP(mdl, truemin)
end

function _namedpoly3_JuMP()
    # the Magnetism polynomial in a ball
    (x, f, dom, truemin) = getpolydata(:magnetism7_ball)
    # WSOS formulation
    mdl = build_JuMP_namedpoly_WSOS(x, f, dom, d=2)
    solveandcheck_JuMP(mdl, truemin)
    # SDP formulation
    # mdl = build_JuMP_namedpoly_PSD(x, f, dom, d=2)
    # solveandcheck_JuMP(mdl, truemin)
end

function _namedpoly4_JuMP()
    # the Motzkin polynomial in an ellipsoid containing two local minima in opposite orthants
    (x, f, dom, truemin) = getpolydata(:motzkin_ellipsoid)
    # WSOS formulation
    mdl = build_JuMP_namedpoly_WSOS(x, f, dom, d=7)
    solveandcheck_JuMP(mdl, truemin)
    # SDP formulation
    # mdl = build_JuMP_namedpoly_PSD(x, f, dom, d=7)
    # solveandcheck_JuMP(mdl, truemin)
end

function _namedpoly5_JuMP()
    (x, f, dom, truemin) = getpolydata(:caprasse)
    # WSOS formulation
    mdl = build_JuMP_namedpoly_WSOS(x, f, dom, d=4)
    solveandcheck_JuMP(mdl, truemin)
    # SDP formulation
    # mdl = build_JuMP_namedpoly_PSD(x, f, dom, d=4)
    # solveandcheck_JuMP(mdl, truemin)
end

function _namedpoly6_JuMP()
    (x, f, dom, truemin) = getpolydata(:goldsteinprice)
    # WSOS formulation
    mdl = build_JuMP_namedpoly_WSOS(x, f, dom, d=7)
    solveandcheck_JuMP(mdl, truemin)
    # SDP formulation
    # mdl = build_JuMP_namedpoly_PSD(x, f, dom, d=7)
    # solveandcheck_JuMP(mdl, truemin)
end

function _namedpoly7_JuMP()
    (x, f, dom, truemin) = getpolydata(:lotkavolterra)
    # WSOS formulation
    mdl = build_JuMP_namedpoly_WSOS(x, f, dom, d=3)
    solveandcheck_JuMP(mdl, truemin)
    # SDP formulation
    # mdl = build_JuMP_namedpoly_PSD(x, f, dom, d=3)
    # solveandcheck_JuMP(mdl, truemin)
end

function _namedpoly8_JuMP()
    (x, f, dom, truemin) = getpolydata(:robinson)
    # WSOS formulation
    mdl = build_JuMP_namedpoly_WSOS(x, f, dom, d=8)
    solveandcheck_JuMP(mdl, truemin)
    # SDP formulation
    # mdl = build_JuMP_namedpoly_PSD(x, f, dom, d=8)
    # solveandcheck_JuMP(mdl, truemin)
end

function _namedpoly9_JuMP()
    (x, f, dom, truemin) = getpolydata(:reactiondiffusion_ball)
    # WSOS formulation
    mdl = build_JuMP_namedpoly_WSOS(x, f, dom, d=3)
    solveandcheck_JuMP(mdl, truemin)
    # SDP formulation
    # mdl = build_JuMP_namedpoly_PSD(x, f, dom, d=3)
    # solveandcheck_JuMP(mdl, truemin)
end

function _namedpoly10_JuMP()
    (x, f, dom, truemin) = getpolydata(:rosenbrock)
    # WSOS formulation
    mdl = build_JuMP_namedpoly_WSOS(x, f, dom, d=4)
    solveandcheck_JuMP(mdl, truemin)
    # SDP formulation
    # mdl = build_JuMP_namedpoly_PSD(x, f, dom, d=4)
    # solveandcheck_JuMP(mdl, truemin)
end

function _shapeconregr1_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 3, 100, 0.0, x -> exp(norm(x)))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (mdl, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=false)
    truemin = 4.4065e-1
    solveandcheck_JuMP(mdl, truemin)
end

function _shapeconregr2_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 3, 100, 0.0, x -> sum(x.^3))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (mdl, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=false)
    truemin = 1.3971e-1
    solveandcheck_JuMP(mdl, truemin)
    # test with non-sampling based interpolation
    (mdl, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=false, sample=false)
    solveandcheck_JuMP(mdl, truemin)
end

function _shapeconregr3_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 3, 100, 0.0, x -> sum(x.^4))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (mdl, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=false)
    truemin = 2.4577e-1
    solveandcheck_JuMP(mdl, truemin)
end

function _shapeconregr4_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 3, 100, 50.0, x -> sum(x.^3))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (mdl, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=false)
    truemin = 1.5449e-1
    solveandcheck_JuMP(mdl, truemin)
end

function _shapeconregr5_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 3, 100, 50.0, x -> sum(x.^4))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (mdl, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=false)
    truemin = 2.5200e-1
    solveandcheck_JuMP(mdl, truemin)
end

function _shapeconregr6_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 3, 100, 0.0, x -> exp(norm(x)))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (mdl, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=true)
    truemin = 5.4584e-2
    solveandcheck_JuMP(mdl, truemin)
end

function _shapeconregr7_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 3, 100, 50.0, x -> sum(x.^4))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (mdl, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=true)
    truemin = 3.3249e-2
    solveandcheck_JuMP(mdl, truemin)
end

function _shapeconregr8_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 4, 100, 0.0, x -> -inv(1 + exp(-10.0 * norm(x))))
    (X, y) = generateregrdata(f, 0.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    shapedata = ShapeData(Hypatia.Box(zeros(n), ones(n)), Hypatia.Box(zeros(n), ones(n)), ones(n), 1)
    (mdl, p) = build_shapeconregr_WSOS(X, y, deg, shapedata, use_leastsqobj=true)
    truemin = 3.7723e-03
    solveandcheck_JuMP(mdl, truemin)
end

function _shapeconregr9_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 4, 100, 10.0, x -> -inv(1 + exp(-10.0 * norm(x))))
    (X, y) = generateregrdata(f, 0.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    shapedata = ShapeData(Hypatia.Box(zeros(n), ones(n)), Hypatia.Box(zeros(n), ones(n)), ones(n), 1)
    (mdl, p) = build_shapeconregr_WSOS(X, y, deg, shapedata, use_leastsqobj=true)
    truemin = 3.0995e-02 # <---- not verified with SDP like others
    solveandcheck_JuMP(mdl, truemin)
end

function _shapeconregr10_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 4, 100, 0.0, x -> exp(norm(x)))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (mdl, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=true)
    truemin = 5.0209e-02 # <---- not verified with SDP like others
    solveandcheck_JuMP(mdl, truemin)
end

function _shapeconregr11_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 5, 100, 10.0, x -> exp(norm(x)))
    (X, y) = generateregrdata(f, 0.5, 2.0, n, npoints, signal_ratio=signal_ratio)
    shapedata = ShapeData(Hypatia.Box(0.5*ones(n), 2*ones(n)), Hypatia.Box(0.5*ones(n), 2*ones(n)), ones(n), 1)
    (mdl, p) = build_shapeconregr_WSOS(X, y, deg, shapedata, use_leastsqobj=true)
    truemin = 0.2283 # <---- not verified with SDP like others
    solveandcheck_JuMP(mdl, truemin)
end

function _shapeconregr12_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 5, 100, 10.0, x -> exp(norm(x)))
    (X, y) = generateregrdata(f, 0.5, 2.0, n, npoints, signal_ratio=signal_ratio)
    shapedata = ShapeData(Hypatia.Box(0.5*ones(n), 2*ones(n)), Hypatia.Box(0.5*ones(n), 2*ones(n)), ones(n), 1)
    (mdl, p) = build_shapeconregr_PSD(X, y, deg, shapedata, use_leastsqobj=true)
    truemin = 0.2283 # <---- not verified with SDP
    solveandcheck_JuMP(mdl, truemin)
end

function _shapeconregr13_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 6, 100, 1.0, x -> exp(norm(x)))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (mdl, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=false)
    truemin = 1.7751 # <---- not verified with SDP like others
    solveandcheck_JuMP(mdl, truemin)
end

function _shapeconregr14_JuMP() # out of memory error when converting sparse to dense in MOI conversion
    (n, deg, npoints, signal_ratio, f) = (5, 5, 1000, 0.0, x -> exp(norm(x)))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (mdl, p) = build_shapeconregr_PSD(X, y, deg, ShapeData(n), use_leastsqobj=true, usedense=true)
    JuMP.optimize!(mdl)
end

function _shapeconregr15_JuMP() # out of memory error during preprocessing
    (n, deg, npoints, signal_ratio, f) = (5, 5, 1000, 0.0, x -> exp(norm(x)))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (mdl, p) = build_shapeconregr_PSD(X, y, deg, ShapeData(n), use_leastsqobj=true, usedense=false)
    JuMP.optimize!(mdl)
end
