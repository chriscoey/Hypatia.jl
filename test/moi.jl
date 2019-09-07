#=
Copyright 2018, Chris Coey and contributors
=#

using Test
import MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIB = MOI.Bridges
const MOIU = MOI.Utilities
import Hypatia
const SO = Hypatia.Solvers

config = MOIT.TestConfig(
    atol = 1e-4,
    rtol = 1e-4,
    solve = true,
    query = true,
    modify_lhs = true,
    duals = true,
    infeas_certificates = true,
    )

unit_exclude = [
    "solve_qcp_edge_cases",
    "solve_qp_edge_cases",
    "solve_integer_edge_cases",
    "solve_objbound_edge_cases",
    "solve_zero_one_with_bounds_1",
    "solve_zero_one_with_bounds_2",
    "solve_zero_one_with_bounds_3",
    "solve_unbounded_model", # dual equalities are inconsistent, so detect dual infeasibility but currently no certificate or status
    ]

conic_exclude = String[
    # "lin",
    # "norminf",
    # "normone",
    # "soc",
    # "rsoc",
    # "exp",
    # "geomean",
    # "pow",
    # "sdp",
    # "logdet",
    # "rootdet",
    # TODO currently some issue with square det transformation?
    "logdets",
    "rootdets",
    ]

function test_moi(
    T::Type{<:Real},
    use_dense::Bool;
    solver_options...
    )
    optimizer = MOIU.CachingOptimizer(MOIU.UniversalFallback(MOIU.Model{T}()),
        Hypatia.Optimizer{T}(use_dense = use_dense; solver_options...))

    @testset "unit tests" begin
        MOIT.unittest(optimizer, config, unit_exclude)
    end
    
    @testset "linear tests" begin
        MOIT.contlineartest(optimizer, config)
    end

    @testset "conic tests" begin
        MOIT.contconictest(MOIB.Constraint.Square{T}(MOIB.Constraint.RootDet{T}(optimizer)), config, conic_exclude)
    end

    return
end
