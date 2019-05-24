#=
Copyright 2018, Chris Coey and contributors

functions and caches for interior point algorithms
=#

module Solvers

using Printf
using LinearAlgebra
using SparseArrays
using Test

import Hypatia.Cones
import Hypatia.Models

using TimerOutputs
import Hypatia

abstract type Solver end

# homogeneous self-dual embedding algorithm
abstract type HSDStepper end
abstract type CombinedHSDSystemSolver end
include("homogeneous_self_dual/solver.jl")
include("homogeneous_self_dual/combined_step/stepper.jl")
include("homogeneous_self_dual/combined_step/naive.jl")
include("homogeneous_self_dual/combined_step/naiveelim.jl")
include("homogeneous_self_dual/combined_step/symindef.jl")
include("homogeneous_self_dual/combined_step/qrchol.jl")
# include("homogeneous_self_dual/combined_step/cholchol.jl")

# TODO sequential quadratic algorithm for linear, quadratic, and smooth convex models

get_status(solver::Solver) = solver.status
get_solve_time(solver::Solver) = solver.solve_time
get_num_iters(solver::Solver) = solver.num_iters

get_primal_obj(solver::Solver) = solver.primal_obj
get_dual_obj(solver::Solver) = solver.dual_obj

get_s(solver::Solver) = copy(solver.point.s)
get_s(solver::Solver, model::Models.Model) = get_s(solver)
get_z(solver::Solver) = copy(solver.point.z)
get_z(solver::Solver, model::Models.Model) = get_z(solver)

get_x(solver::Solver) = copy(solver.point.x)
function get_x(solver::Solver, model::Models.PreprocessedLinearModel)
    x = zeros(length(model.c_raw))
    x[model.x_keep_idxs] = solver.point.x # unpreprocess solver's solution
    return x
end
get_x(solver::Solver, model::Models.Model) = get_x(solver)

get_y(solver::Solver) = copy(solver.point.y)
function get_y(solver::Solver, model::Models.PreprocessedLinearModel)
    y = zeros(length(model.b_raw))
    y[model.y_keep_idxs] = solver.point.y # unpreprocess solver's solution
    return y
end
get_y(solver::Solver, model::Models.Model) = get_y(solver)

# check conic certificates are valid
function get_certificates(
    solver::Solver,
    model::Models.LinearModel;
    test::Bool = true,
    atol::Float64 = 1e-4,
    rtol::Float64 = 1e-4,
    )
    status = get_status(solver)
    primal_obj = get_primal_obj(solver)
    dual_obj = get_dual_obj(solver)
    x = get_x(solver, model)
    y = get_y(solver, model)
    s = get_s(solver, model)
    z = get_z(solver, model)

    if test
        (c, A, b, G, h, cones, cone_idxs) = Models.get_original_data(model)
        if status == :Optimal
            @test primal_obj ≈ dual_obj atol=atol rtol=rtol
            @test A * x ≈ b atol=atol rtol=rtol
            @test G * x + s ≈ h atol=atol rtol=rtol
            @test G' * z + A' * y ≈ -c atol=atol rtol=rtol
            @test dot(s, z) ≈ 0.0 atol=atol rtol=rtol
            @test dot(c, x) ≈ primal_obj atol=1e-8 rtol=1e-8
            @test dot(b, y) + dot(h, z) ≈ -dual_obj atol=1e-8 rtol=1e-8
        elseif status == :PrimalInfeasible
            # @test isnan(primal_obj)
            @test dual_obj > 0
            @test dot(b, y) + dot(h, z) ≈ -dual_obj atol=1e-8 rtol=1e-8
            @test G' * z ≈ -A' * y atol=atol rtol=rtol
        elseif status == :DualInfeasible
            # @test isnan(dual_obj)
            @test primal_obj < 0
            @test dot(c, x) ≈ primal_obj atol=1e-8 rtol=1e-8
            @test G * x ≈ -s atol=atol rtol=rtol
            @test A * x ≈ zeros(length(y)) atol=atol rtol=rtol
        elseif status == :IllPosed
            # TODO primal vs dual ill-posed statuses and conditions
        end
    end

    return (x = x, y = y, s = s, z = z, primal_obj = primal_obj, dual_obj = dual_obj, status = status)
end

end
