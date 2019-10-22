#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

tests using HSL sparse linear system solver caches
requires that HSL.jl be installed and built successfully; Requires.jl handles this optional dependency
=#

import HSL

include(joinpath(@__DIR__, "native.jl"))

const SO = Hypatia.Solvers

testfuns = [
    nonnegative1,
    nonnegative2,
    nonnegative3,
    epinorminf1,
    epinorminf2,
    epinorminf3,
    epinorminf4,
    epinorminf5,
    epinormeucl1,
    epinormeucl2,
    epipersquare1,
    epipersquare2,
    epipersquare3,
    hypoperlog1,
    hypoperlog2,
    hypoperlog3,
    hypoperlog4,
    hypoperlog5,
    hypoperlog6,
    epiperexp1,
    epiperexp2,
    power1,
    power2,
    power3,
    power4,
    hypogeomean1,
    hypogeomean2,
    hypogeomean3,
    epinormspectral1,
    possemideftri1,
    possemideftri2,
    possemideftricomplex1,
    hypoperlogdettri1,
    hypoperlogdettri2,
    hypoperlogdettri3,
    primalinfeas1,
    primalinfeas2,
    primalinfeas3,
    dualinfeas1,
    dualinfeas2,
    dualinfeas3,
    ]

options = (verbose = false,)

@info("starting HSL cache tests")
@testset "HSL cache tests" begin
    @testset "cache setup tests: $cache_type" for cache_type in [Hypatia.HSLSymCache] # TODO wrap and test a HSLNonSymCache
        cache = cache_type()
        @test cache.analyzed == false
        @test isa(cache, cache_type{Float64})
        cache32 = cache_type{Float32}()
        @test cache32.analyzed == false
        @test_throws Exception cache_type{BigFloat}() # TODO make error more specific
    end

    @testset "SymIndefSparse tests: $t" for t in testfuns
        T = Float64
        t(T, solver = SO.Solver{T}(system_solver = SO.SymIndefSparseSystemSolver{T}(fact_cache = Hypatia.HSLSymCache{T}()); options...))
    end
end
