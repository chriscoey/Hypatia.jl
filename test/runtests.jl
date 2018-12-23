#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using Hypatia
using Test
using Random
using LinearAlgebra
using SparseArrays


# load examples functions in src/examples/
egs_dir = joinpath(@__DIR__, "../examples")
include(joinpath(egs_dir, "envelope/native.jl"))
include(joinpath(egs_dir, "envelope/jump.jl"))
include(joinpath(egs_dir, "expdesign/jump.jl"))
include(joinpath(egs_dir, "linearopt/native.jl"))
include(joinpath(egs_dir, "namedpoly/native.jl"))
include(joinpath(egs_dir, "namedpoly/jump.jl"))
include(joinpath(egs_dir, "shapeconregr/jump.jl"))
include(joinpath(egs_dir, "densityest/jump.jl"))
include(joinpath(@__DIR__, "examples.jl"))


# TODO make first part a native interface function eventually
# TODO maybe build a new high-level model struct; the current model struct is low-level
function solveandcheck(
    mdl::Hypatia.Model,
    c,
    A,
    b,
    G,
    h,
    cone,
    lscachetype;
    atol = 1e-4,
    rtol = 1e-4,
    )
    # check, preprocess, load, and solve
    Hypatia.check_data(c, A, b, G, h, cone)
    if lscachetype == Hypatia.QRChol
        (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c, A, b, G, useQR=true)
        L = Hypatia.QRChol(c1, A1, b1, G1, h, cone, Q2, RiQ1)
    elseif lscachetype == Hypatia.Naive3
        (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c, A, b, G, useQR=false)
        L = Hypatia.Naive3(c1, A1, b1, G1, h, cone)
    else
        error("linear system cache type $lscachetype is not recognized")
    end
    Hypatia.load_data!(mdl, c1, A1, b1, G1, h, cone, L)
    Hypatia.solve!(mdl)

    # construct solution
    x = zeros(length(c))
    x[dukeep] = Hypatia.get_x(mdl)
    y = zeros(length(b))
    y[prkeep] = Hypatia.get_y(mdl)
    s = Hypatia.get_s(mdl)
    z = Hypatia.get_z(mdl)
    pobj = Hypatia.get_pobj(mdl)
    dobj = Hypatia.get_dobj(mdl)
    status = Hypatia.get_status(mdl)

    # check conic certificates are valid; conditions are described by CVXOPT at https://github.com/cvxopt/cvxopt/blob/master/src/python/coneprog.py
    Hypatia.loadpnt!(cone, s, z)
    if status == :Optimal
        # @test Hypatia.incone(cone)
        @test pobj ≈ dobj atol=atol rtol=rtol
        @test A*x ≈ b atol=atol rtol=rtol
        @test G*x + s ≈ h atol=atol rtol=rtol
        @test G'*z + A'*y ≈ -c atol=atol rtol=rtol
        @test dot(s, z) ≈ 0.0 atol=atol rtol=rtol
        @test dot(c, x) ≈ pobj atol=1e-8 rtol=1e-8
        @test dot(b, y) + dot(h, z) ≈ -dobj atol=1e-8 rtol=1e-8
    elseif status == :PrimalInfeasible
        # @test Hypatia.incone(cone)
        @test isnan(pobj)
        @test dobj > 0
        @test dot(b, y) + dot(h, z) ≈ -dobj atol=1e-8 rtol=1e-8
        @test G'*z ≈ -A'*y atol=atol rtol=rtol
    elseif status == :DualInfeasible
        # @test Hypatia.incone(cone)
        @test isnan(dobj)
        @test pobj < 0
        @test dot(c, x) ≈ pobj atol=1e-8 rtol=1e-8
        @test G*x ≈ -s atol=atol rtol=rtol
        @test A*x ≈ zeros(length(y)) atol=atol rtol=rtol
    elseif status == :IllPosed
        # @test Hypatia.incone(cone)
        # TODO primal vs dual ill-posed statuses and conditions
    end

    stime = Hypatia.get_solvetime(mdl)
    niters = Hypatia.get_niters(mdl)

    return (x=x, y=y, s=s, z=z, pobj=pobj, dobj=dobj, status=status, stime=stime, niters=niters)
end


@testset begin

@info("starting interpolation tests")
include(joinpath(@__DIR__, "interpolation.jl"))


include(joinpath(@__DIR__, "native.jl"))
@info("starting native interface tests")
verbose = false
lscachetypes = [
    Hypatia.QRChol,
    Hypatia.Naive3,
    ]
testfuns = [
    _dimension1,
    _consistent1,
    _inconsistent1,
    _inconsistent2,
    _orthant1,
    _orthant2,
    _orthant3,
    _orthant4,
    _epinorminf1,
    _epinorminf2,
    _epinorminf3,
    _epinorminf4,
    _epinorminf5,
    _epinorminf6,
    _epinormeucl1,
    _epinormeucl2,
    _epipersquare1,
    _epipersquare2,
    _epipersquare3,
    _semidefinite1,
    _semidefinite2,
    _semidefinite3,
    _hypoperlog1,
    _hypoperlog2,
    _hypoperlog3,
    _hypoperlog4,
    _epiperpower1,
    _epiperpower2,
    _epiperpower3,
    _hypogeomean1,
    _hypogeomean2,
    _hypogeomean3,
    _hypogeomean4,
    _epinormspectral1,
    _hypoperlogdet1,
    _hypoperlogdet2,
    _hypoperlogdet3,
    _epipersumexp1,
    _epipersumexp2,
    ]
@testset "native tests: $testfun, $lscachetype" for testfun in testfuns, lscachetype in lscachetypes
    testfun(verbose=verbose, lscachetype=lscachetype)
end


@info("starting native examples tests")
verbose = false
lscachetypes = [
    Hypatia.QRChol,
    # Hypatia.Naive3, # slow
    ]
testfuns = [
    _envelope1,
    _envelope2,
    _envelope3,
    _envelope4,
    _linearopt1,
    _linearopt2,
    _namedpoly1,
    _namedpoly2,
    _namedpoly3,
    # _namedpoly4, # interpolation memory usage excessive
    _namedpoly5,
    # _namedpoly6, # interpolation memory usage excessive
    _namedpoly7,
    _namedpoly8,
    _namedpoly9,
    _namedpoly10, # numerically unstable
    _namedpoly11,
    ]
@testset "native examples: $testfun, $lscachetype" for testfun in testfuns, lscachetype in lscachetypes
    testfun(verbose=verbose, lscachetype=lscachetype)
end


@info("starting JuMP examples tests")
testfuns = [
    _namedpoly1_JuMP,
    _namedpoly2_JuMP,
    _namedpoly3_JuMP,
    _namedpoly4_JuMP, # numerically unstable
    _namedpoly5_JuMP,
    _namedpoly6_JuMP,
    _namedpoly7_JuMP,
    _namedpoly8_JuMP,
    _namedpoly9_JuMP,
    _namedpoly10_JuMP,
    _shapeconregr1_JuMP,
    _shapeconregr2_JuMP,
    _shapeconregr3_JuMP,
    _shapeconregr4_JuMP,
    _shapeconregr5_JuMP,
    _shapeconregr6_JuMP,
    _shapeconregr7_JuMP, # numerically unstable
    _shapeconregr8_JuMP,
    _shapeconregr9_JuMP, # numerically unstable
    _shapeconregr10_JuMP, # numerically unstable
    _shapeconregr11_JuMP, # numerically unstable
    _shapeconregr12_JuMP, # numerically unstable
    _shapeconregr13_JuMP, # numerically unstable
    # _shapeconregr14_JuMP, # throws out-of-memory error
    # _shapeconregr15_JuMP, # throws out-of-memory error
    ]
@testset "JuMP examples: $testfun" for testfun in testfuns
    testfun()
end


@info("starting verbose default examples tests")
testfuns = [
    run_JuMP_expdesign,
    run_linearopt,
    run_namedpoly,
    run_JuMP_namedpoly_PSD, # final objective doesn't match
    run_JuMP_namedpoly_WSOS_primal,
    run_JuMP_namedpoly_WSOS_dual,
    run_envelope_primal_dense,
    run_envelope_dual_dense,
    run_envelope_primal_sparse,
    run_envelope_dual_sparse,
    run_JuMP_envelope_boxinterp,
    run_JuMP_envelope_sampleinterp_box,
    run_JuMP_envelope_sampleinterp_ball,
    run_JuMP_shapeconregr_PSD,
    run_JuMP_shapeconregr_WSOS,
    run_JuMP_densityest,
    ]
@testset "default examples: $testfun" for testfun in testfuns
    testfun()
end


@info("starting MathOptInterface tests")
include(joinpath(@__DIR__, "MOI_wrapper.jl"))
verbose = false
lscachetypes = [
    Hypatia.QRChol,
    Hypatia.Naive3,
    ]
@testset "MOI tests: $lscachetype, $(usedense ? "dense" : "sparse")" for lscachetype in lscachetypes, usedense in [false, true]
    testmoi(verbose=verbose, lscachetype=lscachetype, usedense=usedense)
end

end
