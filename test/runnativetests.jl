#=
Copyright 2019, Chris Coey and contributors
=#

include(joinpath(@__DIR__, "native.jl"))

const SO = Hypatia.Solvers

testfuns_no_preproc = [
    nonnegative1,
    epinorminf1,
    epinormeucl1,
    epipersquare1,
    hypoperlog1,
    epiperexp1,
    power1,
    hypogeomean1,
    epinormspectral1,
    possemideftri1,
    possemideftricomplex1,
    hypoperlogdettri1,
    primalinfeas1,
    dualinfeas1,
    ]

testfuns_preproc = [
    dimension1,
    consistent1,
    inconsistent1,
    inconsistent2,
    ]

testfuns_reduce = vcat(testfuns_no_preproc, testfuns_preproc)

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
    epinormeucl3,
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

generic_reals = [
    Float64,
    # Float32,
    # BigFloat,
    ]

blas_reals = [
    Float64,
    Float32,
    ]

options = (verbose = true,)

@info("starting native tests")
@testset "native tests" begin
    # # test with and without preprocessing
    # @testset "no preprocessing tests: $t, $T" for t in testfuns_no_preproc, T in generic_reals
    #     t(T, solver = SO.Solver{T}(preprocess = false, init_use_indirect = false, reduce = false, system_solver = SO.SymIndefDenseSystemSolver{T}(); options...)) # TODO make default system solver depend on preprocess
    # end
    # @testset "preprocessing tests: $t, $T" for t in testfuns_preproc, T in generic_reals
    #     t(T, solver = SO.Solver{T}(preprocess = true, init_use_indirect = false, reduce = false; options...))
    # end
    #
    # # test indirect initial point method
    # @testset "indirect initialization tests: $t, $T" for t in testfuns_no_preproc, T in blas_reals
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
    # @testset "neighborhood function tests: $t, $T, $n" for t in testfuns, T in generic_reals, n in [true, false]
    #     t(T, solver = SO.Solver{T}(use_infty_nbhd = n; options...))
    # end

    # test each system solver
    # @testset "NaiveDense tests: $t, $T" for t in testfuns, T in generic_reals
    #     t(T, solver = SO.Solver{T}(reduce = true, system_solver = SO.NaiveDenseSystemSolver{T}(); options...))
    #     # t(T, solver = SO.Solver{T}(reduce = false, system_solver = SO.NaiveDenseSystemSolver{T}(); options...))
    # end
    # @testset "NaiveSparse tests: $t" for t in testfuns
    #     T = Float64
    #     t(T, solver = SO.Solver{T}(system_solver = SO.NaiveSparseSystemSolver{T}(); options...))
    # end
    # @testset "NaiveIndirect tests: $t" for t in testfuns # TODO need to use linearmaps here with the apply_LHS function, not blockmatrix
    #     T = Float64
    #     t(T, solver = SO.Solver{T}(preprocess = false, init_use_indirect = true, reduce = false, system_solver = SO.NaiveIndirectSystemSolver{T}(); options...))
    # end
    # @testset "NaiveElimDense tests: $t, $T, $h" for t in testfuns, T in generic_reals, h in [true, false]
    #     t(T, solver = SO.Solver{T}(system_solver = SO.NaiveElimDenseSystemSolver{T}(use_inv_hess = h); options...))
    # end
    # @testset "NaiveElimSparse tests: $t" for t in testfuns
    #     T = Float64
    #     t(T, solver = SO.Solver{T}(system_solver = SO.NaiveElimSparseSystemSolver{T}(); options...))
    # end
    @testset "SymIndefDense tests: $t, $T" for t in testfuns, T in generic_reals
        t(T, solver = SO.Solver{T}(system_solver = SO.SymIndefDenseSystemSolver{T}(); options...))
    end
    @testset "SymIndefSparse tests: $t" for t in testfuns
        T = Float64
        t(T, solver = SO.Solver{T}(system_solver = SO.SymIndefSparseSystemSolver{T}(); options...))
    end
    # @testset "QRCholDense tests: $t, $T, $ss" for t in testfuns, T in generic_reals, ss in [Hypatia.DenseSymCache, Hypatia.DensePosDefCache]
    #     t(T, solver = SO.Solver{T}(system_solver = SO.QRCholDenseSystemSolver{T}(fact_cache = ss{T}()); options...))
    # end
end
