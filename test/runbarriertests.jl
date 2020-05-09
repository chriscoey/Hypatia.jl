#=
Copyright 2019, Chris Coey and contributors
=#

include(joinpath(@__DIR__, "barrier.jl"))

barrier_testfuns = [
    test_nonnegative_barrier,
    # test_epinorminf_barrier,
    # test_epinormeucl_barrier,
    # test_epipersquare_barrier,
    test_hypoperlog_barrier,
    # test_episumperentropy_barrier,
    # test_power_barrier,
    # test_hypogeomean_barrier,
    # test_epinormspectral_barrier,
    # test_matrixepipersquare_barrier,
    # test_linmatrixineq_barrier,
    # test_possemideftri_barrier,
    # test_possemideftrisparse_barrier,
    # test_hypoperlogdettri_barrier,
    # test_hyporootdettri_barrier,
    # test_wsosinterpnonnegative_barrier,
    # test_wsosinterppossemideftri_barrier,
    # test_wsosinterpepinormeucl_barrier,
    ]

real_types = [
    Float64,
    Float32,
    BigFloat,
    ]

@info("starting barrier tests")
@testset "barrier tests: $t, $T" for t in barrier_testfuns, T in real_types
    t(T)
end
