#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

tests using HSL sparse linear system solver caches
requires that HSL.jl be installed and built successfully; Requires.jl handles this optional dependency
=#

import HSL

include(joinpath(@__DIR__, "nativeinstances.jl"))
include(joinpath(@__DIR__, "nativesets.jl"))

const SO = Hypatia.Solvers

options = (verbose = false,)

@info("starting HSL cache tests")
@testset "HSL cache tests" begin
    @testset "cache setup tests: $cache_type" for cache_type in [Hypatia.HSLSymCache] # TODO wrap and test a HSLNonSymCache
        cache = cache_type()
        @test isa(cache, cache_type{Float64})
        @test cache.analyzed == false
        @test cache_type{Float32}().analyzed == false
        @test_throws Exception cache_type{BigFloat}() # TODO make error more specific
    end

    @testset "SymIndefSparse tests: $t, $T" for t in testfuns_many, T in [Float32, Float64]
        t(T, solver = SO.Solver{T}(system_solver = SO.SymIndefSparseSystemSolver{T}(fact_cache = Hypatia.HSLSymCache{T}()); options...))
    end
end
