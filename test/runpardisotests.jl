#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

tests using Pardiso sparse linear system solver caches
requires that Pardiso.jl be installed and built successfully; Requires.jl handles this optional dependency
=#

import Pardiso

include(joinpath(@__DIR__, "nativeinstances.jl"))
include(joinpath(@__DIR__, "nativesets.jl"))

const SO = Hypatia.Solvers

options = (verbose = false,)

@info("starting Pardiso cache tests")
@testset "Pardiso cache tests" begin
    @testset "cache setup tests: $cache_type" for cache_type in [Hypatia.PardisoSymCache, Hypatia.PardisoNonSymCache]
        cache = cache_type()
        @test isa(cache, cache_type{Float64})
        @test cache.analyzed == false
        @test_throws Exception cache_type{Float32}() # TODO make error more specific
    end

    T = Float64
    @testset "NaiveSparse tests: $t" for t in testfuns
        t(T, solver = SO.Solver{T}(system_solver = SO.NaiveSparseSystemSolver{T}(fact_cache = Hypatia.PardisoNonSymCache()); options...))
    end
    @testset "NaiveElimSparse tests: $t" for t in testfuns
        t(T, solver = SO.Solver{T}(system_solver = SO.NaiveElimSparseSystemSolver{T}(fact_cache = Hypatia.PardisoNonSymCache()); options...))
    end
    @testset "SymIndefSparse tests: $t" for t in testfuns
        t(T, solver = SO.Solver{T}(system_solver = SO.SymIndefSparseSystemSolver{T}(fact_cache = Hypatia.PardisoSymCache()); options...))
    end
end
