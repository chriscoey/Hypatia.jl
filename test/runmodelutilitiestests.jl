#=
Copyright 2019, Chris Coey and contributors
=#

include(joinpath(@__DIR__, "modelutilities.jl"))

generic_reals = [
    Float64,
    Float32,
    BigFloat,
    ]

@info("starting interpolation tests")
@testset "interpolation tests: $T" for T in generic_reals
    fekete_sample(T)
    test_recover_lagrange_polys(T)
    test_recover_cheb_polys(T)
end

@info("starting miscellaneous tests")
@testset "misc tests: $T" for T in generic_reals
    test_svec_conversion(T)
end
