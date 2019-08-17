#=
Copyright 2019, Chris Coey and contributors
=#

include(joinpath(@__DIR__, "moi.jl"))

@info("starting MOI tests")

verbose = false

real_types = [
    Float64,
    # BigFloat, # TODO test this when MOI allows
    ]

dense_options = [
    true,
    false,
    ]

system_solvers = [
    SO.NaiveElimCombinedHSDSystemSolver,
    SO.QRCholCombinedHSDSystemSolver,
    ]

@testset "MOI tests: $(d ? "dense" : "sparse"), $s, $T" for d in dense_options, s in system_solvers, T in real_types
    test_moi(d, s{T}, MO.PreprocessedLinearModel{T}, verbose)
end
