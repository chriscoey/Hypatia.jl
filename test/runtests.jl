#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

import Random
using LinearAlgebra
using SparseArrays
using Test
import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MO = HYP.Models
const SO = HYP.Solvers
const MU = HYP.ModelUtilities

include(joinpath(@__DIR__, "interpolation.jl"))
include(joinpath(@__DIR__, "barriers.jl"))
include(joinpath(@__DIR__, "native.jl"))
# include(joinpath(@__DIR__, "MathOptInterface.jl"))

examples_dir = joinpath(@__DIR__, "../examples")
include(joinpath(examples_dir, "envelope/native.jl"))
include(joinpath(examples_dir, "linearopt/native.jl"))
include(joinpath(examples_dir, "polymin/native.jl"))
include(joinpath(examples_dir, "contraction/JuMP.jl"))
include(joinpath(examples_dir, "densityest/JuMP.jl"))
include(joinpath(examples_dir, "envelope/JuMP.jl"))
include(joinpath(examples_dir, "expdesign/JuMP.jl"))
include(joinpath(examples_dir, "lotkavolterra/JuMP.jl"))
include(joinpath(examples_dir, "muconvexity/JuMP.jl"))
include(joinpath(examples_dir, "polymin/JuMP.jl"))
include(joinpath(examples_dir, "polynorm/JuMP.jl"))
include(joinpath(examples_dir, "regionofattr/JuMP.jl"))
include(joinpath(examples_dir, "secondorderpoly/JuMP.jl"))
include(joinpath(examples_dir, "shapeconregr/JuMP.jl"))
include(joinpath(examples_dir, "semidefinitepoly/JuMP.jl"))

@testset "Hypatia tests" begin
#
# @info("starting interpolation tests")
# @testset "interpolation tests" begin
#     fekete_sample()
#     test_recover_lagrange_polys()
#     test_recover_cheb_polys()
# end
#
# @info("starting barrier tests")
# barrier_testfuns = [
#     test_epinormeucl_barrier,
#     test_epinorinf_barrier,
#     test_epinormspectral_barrier,
#     test_epiperpower_barrier,
#     test_epipersquare_barrier,
#     test_epipersumexp_barrier,
#     test_hypogeomean_barrier,
#     test_hypoperlog_barrier,
#     test_hypoperlogdet_barrier,
#     test_semidefinite_barrier,
#     test_wsospolyinterp_barrier,
#     test_wsospolyinterpmat_barrier,
#     test_wsospolyinterpsoc_barrier,
#     ]
# @testset "barrier functions tests: $t" for t in barrier_testfuns
#     t()
# end

@info("starting native interface tests")
verbose = true
system_solvers = [
    SO.NaiveCombinedHSDSystemSolver,
    SO.QRCholCombinedHSDSystemSolver,
    ]
testfuns_singular = [
    dimension1,
    consistent1,
    inconsistent1,
    inconsistent2,
    ]
@testset "preprocessing tests: $t, $s" for t in testfuns_singular, s in system_solvers
    t(s, MO.PreprocessedLinearModel, verbose)
end
linear_models = [
    MO.RawLinearModel,
    MO.PreprocessedLinearModel,
    ]
testfuns_nonsingular = [
    orthant1,
    orthant2,
    orthant3,
    orthant4,
    epinorminf1,
    epinorminf2,
    epinorminf3,
    epinorminf4,
    epinorminf5,
    epinorminf6,
    epinormeucl1,
    epinormeucl2,
    epipersquare1,
    epipersquare2,
    epipersquare3,
    semidefinite1,
    semidefinite2,
    semidefinite3,
    semidefinitecomplex1,
    hypoperlog1,
    hypoperlog2,
    hypoperlog3,
    hypoperlog4,
    epiperpower1,
    epiperpower2,
    epiperpower3,
    hypogeomean1,
    hypogeomean2,
    hypogeomean3,
    hypogeomean4,
    epinormspectral1,
    hypoperlogdet1,
    hypoperlogdet2,
    hypoperlogdet3,
    epipersumexp1,
    epipersumexp2,
    ]
@testset "native tests: $t, $s, $m" for t in testfuns_nonsingular, s in system_solvers, m in linear_models
    if s == SO.QRCholCombinedHSDSystemSolver && m == MO.RawLinearModel
        continue # QRChol linear system solver needs preprocessed model
    end
    t(s, m, verbose)
end

# @info("starting MathOptInterface tests")
# verbose = false
# system_solvers = [
#     SO.NaiveCombinedHSDSystemSolver,
#     SO.QRCholCombinedHSDSystemSolver,
#     ]
# linear_models = [
#     MO.PreprocessedLinearModel, # MOI tests require preprocessing
#     ]
# @testset "MOI tests: $(d ? "dense" : "sparse"), $s, $m" for d in (false, true), s in system_solvers, m in linear_models
#     test_moi(d, s, m, verbose)
# end
#
# @info("starting native examples tests")
# native_options = (
#     verbose = true,
#     max_iters = 150,
#     time_limit = 6e2, # 1 minute
#     )
# @testset "native examples" begin
#     @testset "envelope" begin test_envelope(; native_options...,
#         ) end
#     @testset "linearopt" begin test_linearopt(; native_options...,
#         ) end
#     @testset "polymin" begin test_polymin(; native_options...,
#         tol_rel_opt = 1e-9, tol_abs_opt = 1e-8, tol_feas = 1e-9,
#         ) end
# end
#
# @info("starting JuMP examples tests")
# JuMP_options = (
#     verbose = true,
#     test_certificates = true,
#     max_iters = 250,
#     time_limit = 6e2, # 1 minute
#     )
# @testset "JuMP examples" begin
#     @testset "contraction" begin test_contractionJuMP(; JuMP_options...,
#         tol_rel_opt = 1e-4, tol_abs_opt = 1e-4, tol_feas = 1e-4,
#         ) end
#     @testset "densityest" begin test_densityestJuMP(; JuMP_options...,
#         tol_rel_opt = 1e-5, tol_abs_opt = 1e-5, tol_feas = 1e-6,
#         ) end
#     @testset "envelope" begin test_envelopeJuMP(; JuMP_options...,
#         ) end
#     @testset "expdesign" begin test_expdesignJuMP(; JuMP_options...,
#         ) end
#     @testset "lotkavolterra" begin test_lotkavolterraJuMP(; JuMP_options...,
#         tol_rel_opt = 1e-5, tol_abs_opt = 1e-6, tol_feas = 1e-6,
#         ) end
#     @testset "muconvexity" begin test_muconvexityJuMP(; JuMP_options...,
#         ) end
#     @testset "polymin" begin test_polyminJuMP(; JuMP_options...,
#         tol_rel_opt = 1e-9, tol_abs_opt = 1e-8, tol_feas = 1e-9,
#         ) end
#     @testset "polynorm" begin test_polynormJuMP(; JuMP_options...,
#         ) end
#     @testset "regionofattr" begin test_regionofattrJuMP(; JuMP_options...,
#         tol_abs_opt = 1e-6, tol_rel_opt = 1e-6, tol_feas = 1e-6,
#         ) end
#     @testset "secondorderpoly" begin test_secondorderpolyJuMP(; JuMP_options...,
#         ) end
#     @testset "semidefinitepoly" begin test_semidefinitepolyJuMP(; JuMP_options...,
#         tol_abs_opt = 1e-7, tol_rel_opt = 1e-7, tol_feas = 1e-7,
#         ) end
#     @testset "shapeconregr" begin test_shapeconregrJuMP(; JuMP_options...,
#         tol_rel_opt = 1e-6, tol_abs_opt = 1e-6, tol_feas = 1e-6,
#         ) end
# end

end
