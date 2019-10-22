#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

tests using Pardiso sparse linear system solver caches
requires that Pardiso.jl be installed and built successfully; Requires.jl handles this optional dependency
=#

import Pardiso

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

@info("starting Pardiso cache tests")
@testset "Pardiso cache tests" begin
    @testset "cache setup tests: $cache_type" for cache_type in [Hypatia.PardisoSymCache, Hypatia.PardisoNonSymCache]
        cache = cache_type()
        @test cache.analyzed == false
        @test isa(cache, cache_type{Float64})
        @test_throws Exception cache_type{Float32}() # TODO make error more specific
    end

    @testset "NaiveSparse tests: $t" for t in testfuns
        T = Float64
        t(T, solver = SO.Solver{T}(system_solver = SO.NaiveSparseSystemSolver{T}(fact_cache = Hypatia.PardisoNonSymCache()); options...))
    end

    @testset "NaiveElimSparse tests: $t" for t in testfuns
        T = Float64
        t(T, solver = SO.Solver{T}(system_solver = SO.NaiveElimSparseSystemSolver{T}(fact_cache = Hypatia.PardisoNonSymCache()); options...))
    end

    @testset "SymIndefSparse tests: $t" for t in testfuns
        T = Float64
        t(T, solver = SO.Solver{T}(system_solver = SO.SymIndefSparseSystemSolver{T}(fact_cache = Hypatia.PardisoSymCache()); options...))
    end
end
