#=
Copyright 2019, Chris Coey and contributors
=#

include(joinpath(@__DIR__, "modelutilities.jl"))

@info("starting interpolation tests")
@testset "interpolation tests" begin
    fekete_sample()
    test_recover_lagrange_polys()
    test_recover_cheb_polys()
    test_svec_conversion()
end
