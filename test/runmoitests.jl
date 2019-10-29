#=
Copyright 2019, Chris Coey and contributors
=#

include(joinpath(@__DIR__, "moi.jl"))

system_solvers = [
    SO.NaiveDenseSystemSolver,
    SO.NaiveSparseSystemSolver,
    # SO.NaiveIndirectSystemSolver, # TODO uncomment
    SO.NaiveElimDenseSystemSolver,
    SO.NaiveElimSparseSystemSolver,
    SO.SymIndefDenseSystemSolver,
    SO.SymIndefSparseSystemSolver,
    SO.QRCholDenseSystemSolver,
    ]

real_types = [
    Float64,
    # BigFloat, # TODO test this when MOI allows
    ]

@info("starting MOI tests")
@testset "MOI tests" begin
    @testset "MOI tests: $s, $T" for s in system_solvers, T in real_types
        test_moi(T, verbose = false, system_solver = s{T}())
    end
end
