#=
Copyright 2019, Chris Coey and contributors

TODO
- add native test problems for WSOS cones
- add at least 3 test problems per cone
- speed up tests
- test more cone options eg scaling, 3rd order corrector
=#

include(joinpath(@__DIR__, "nativeinstances.jl"))
include(joinpath(@__DIR__, "nativesets.jl"))

const SO = Hypatia.Solvers

generic_reals = [
    Float64,
    Float32,
    BigFloat,
    ]

options = (verbose = true,)

@info("starting native tests")
@testset "native tests" begin
    # # test with and without preprocessing
    # @testset "no preprocessing tests: $t, $T" for t in testfuns_few, T in generic_reals
    #     t(T, solver = SO.Solver{T}(preprocess = false, init_use_indirect = false, reduce = false, system_solver = SO.SymIndefDenseSystemSolver{T}(); options...)) # TODO make default system solver depend on preprocess
    # end
    # @testset "preprocessing tests: $t, $T" for t in testfuns_preproc, T in generic_reals
    #     t(T, solver = SO.Solver{T}(preprocess = true, init_use_indirect = false, reduce = false; options...))
    # end
    #
    # # test indirect initial point method
    # @testset "indirect initialization tests: $t, $T" for t in testfuns_few, T in generic_reals
    #     T <: LinearAlgebra.BlasReal || continue
    #     t(T, solver = SO.Solver{T}(preprocess = false, init_use_indirect = true, reduce = false, system_solver = SO.SymIndefDenseSystemSolver{T}(); options...))
    # end
    #
    # # test with reduction (removing all primal equalities)
    # @testset "no reduction tests: $t, $T" for t in testfuns_reduce, T in generic_reals
    #     t(T, solver = SO.Solver{T}(preprocess = true, init_use_indirect = false, reduce = false, system_solver = SO.QRCholDenseSystemSolver{T}(); options...))
    # end
    # @testset "reduction tests: $t, $T" for t in testfuns_reduce, T in generic_reals
    #     t(T, solver = SO.Solver{T}(preprocess = true, init_use_indirect = false, reduce = true, system_solver = SO.QRCholDenseSystemSolver{T}(); options...))
    # end
    #
    # # test with different neighborhood functions
    # @testset "neighborhood function tests: $t, $T, $n" for t in testfuns_many, T in generic_reals, n in [true, false]
    #     t(T, solver = SO.Solver{T}(use_infty_nbhd = n; options...))
    # end

    # # test each system solver
    @testset "NaiveDense tests: $t, $T" for t in testfuns_many, T in generic_reals
        t(T, solver = SO.Solver{T}(system_solver = SO.NaiveDenseSystemSolver{T}(); options...))
    end
    # @testset "NaiveSparse tests: $t" for t in testfuns_many
    #     T = Float64
    #     t(T, solver = SO.Solver{T}(system_solver = SO.NaiveSparseSystemSolver{T}(); options...))
    # end
    # @testset "NaiveElimDense tests: $t, $T, $h" for t in testfuns_many, T in generic_reals, h in [true, false]
    #     t(T, solver = SO.Solver{T}(system_solver = SO.NaiveElimDenseSystemSolver{T}(use_inv_hess = h); options...))
    # end
    # @testset "NaiveElimSparse tests: $t" for t in testfuns_many
    #     T = Float64
    #     t(T, solver = SO.Solver{T}(reduce = false, system_solver = SO.NaiveElimSparseSystemSolver{T}(); options...))
    # end
    # @testset "SymIndefDense tests: $t, $T" for t in testfuns_many, T in generic_reals
    #     t(T, solver = SO.Solver{T}(system_solver = SO.SymIndefDenseSystemSolver{T}(); options...))
    # end
    # @testset "SymIndefSparse tests: $t" for t in testfuns_many
    #     T = Float64
    #     t(T, solver = SO.Solver{T}(system_solver = SO.SymIndefSparseSystemSolver{T}(); options...))
    # end
    # @testset "QRCholDense tests: $t, $T, $ss" for t in testfuns_many, T in generic_reals, ss in [Hypatia.DenseSymCache, Hypatia.DensePosDefCache]
    #     t(T, solver = SO.Solver{T}(system_solver = SO.QRCholDenseSystemSolver{T}(fact_cache = ss{T}()); options...))
    # end
end
