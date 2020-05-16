#=
Copyright 2018, Chris Coey and contributors

functions and caches for interior point algorithms
=#

module Solvers

using Printf
using LinearAlgebra
using SparseArrays
import SuiteSparse
import IterativeSolvers
using Test
using TimerOutputs
import Hypatia.Cones
import Hypatia.Models
import Hypatia.BlockMatrix
import Hypatia.SparseNonSymCache
import Hypatia.SparseSymCache
import Hypatia.update_fact
import Hypatia.inv_prod
import Hypatia.free_memory
import Hypatia.int_type
import Hypatia.DenseNonSymCache
import Hypatia.DenseSymCache
import Hypatia.DensePosDefCache
import Hypatia.load_matrix
import Hypatia.invert

default_tol(::Type{T}) where {T <: Real} = sqrt(eps(T))
default_tol(::Type{BigFloat}) = eps(BigFloat) ^ 0.4

abstract type Stepper{T <: Real} end

abstract type SystemSolver{T <: Real} end

mutable struct Solver{T <: Real}
    # main options
    verbose::Bool
    iter_limit::Int
    time_limit::Float64
    tol_rel_opt::T
    tol_abs_opt::T
    tol_feas::T
    tol_slow::T
    preprocess::Bool
    reduce::Bool
    init_use_indirect::Bool
    init_tol_qr::T
    init_use_fallback::Bool
    max_nbhd::T
    stepper::Stepper{T}
    system_solver::SystemSolver{T}
    timer::TimerOutput

    # current status of the solver object and info
    status::Symbol
    solve_time::Float64
    num_iters::Int

    # model and preprocessed model data
    orig_model::Models.Model{T}
    model::Models.Model{T}
    x_keep_idxs::AbstractVector{Int}
    y_keep_idxs::AbstractVector{Int}
    Ap_R::UpperTriangular{T, <:AbstractMatrix{T}}
    Ap_Q::Union{UniformScaling, AbstractMatrix{T}}
    reduce_cQ1
    reduce_Rpib0
    reduce_GQ1
    reduce_Ap_R
    reduce_Ap_Q
    reduce_y_keep_idxs
    reduce_row_piv_inv

    # current iterate
    point::Models.Point{T}
    tau::T
    kap::T
    mu::T

    # residuals
    x_residual::Vector{T}
    y_residual::Vector{T}
    z_residual::Vector{T}
    x_norm_res_t::T
    y_norm_res_t::T
    z_norm_res_t::T
    x_norm_res::T
    y_norm_res::T
    z_norm_res::T

    # convergence parameters
    primal_obj_t::T
    dual_obj_t::T
    primal_obj::T
    dual_obj::T
    gap::T
    rel_gap::T
    x_feas::T
    y_feas::T
    z_feas::T

    # termination condition helpers
    x_conv_tol::T
    y_conv_tol::T
    z_conv_tol::T
    prev_is_slow::Bool
    prev2_is_slow::Bool
    prev_gap::T
    prev_rel_gap::T
    prev_x_feas::T
    prev_y_feas::T
    prev_z_feas::T

    function Solver{T}(;
        verbose::Bool = true,
        iter_limit::Int = 1000,
        time_limit::Real = Inf,
        tol_rel_opt::Real = default_tol(T),
        tol_abs_opt::Real = default_tol(T),
        tol_feas::Real = default_tol(T),
        tol_slow::Real = 1e-3,
        preprocess::Bool = true,
        reduce::Bool = true,
        init_use_indirect::Bool = false,
        init_tol_qr::Real = 1000 * eps(T),
        init_use_fallback::Bool = true,
        max_nbhd::Real = Cones.default_max_neighborhood(), # TODO cleanup - only for taukap, maybe use full name
        stepper::Stepper{T} = CombinedStepper{T}(),
        system_solver::SystemSolver{T} = QRCholDenseSystemSolver{T}(),
        timer::TimerOutput = TimerOutput(),
        ) where {T <: Real}
        if isa(system_solver, QRCholSystemSolver{T})
            @assert preprocess # require preprocessing for QRCholSystemSolver # TODO only need primal eq preprocessing or reduction
        end
        if reduce
            @assert preprocess # cannot use reduction without preprocessing # TODO only need primal eq preprocessing
        end
        @assert !(init_use_indirect && preprocess) # cannot use preprocessing and indirect methods for initial point

        solver = new{T}()

        solver.verbose = verbose
        solver.iter_limit = iter_limit
        solver.time_limit = time_limit
        solver.tol_rel_opt = tol_rel_opt
        solver.tol_abs_opt = tol_abs_opt
        solver.tol_feas = tol_feas
        solver.tol_slow = tol_slow
        solver.preprocess = preprocess
        solver.reduce = reduce
        solver.init_use_indirect = init_use_indirect
        solver.init_tol_qr = init_tol_qr
        solver.init_use_fallback = init_use_fallback
        solver.max_nbhd = max_nbhd
        solver.stepper = stepper
        solver.system_solver = system_solver
        solver.timer = timer
        solver.status = :NotLoaded

        return solver
    end
end

function solve(solver::Solver{T}) where {T <: Real}
    @assert solver.status == :Loaded
    solver.status = :SolveCalled
    start_time = time()
    solver.num_iters = 0
    solver.solve_time = NaN

    solver.x_norm_res_t = NaN
    solver.y_norm_res_t = NaN
    solver.z_norm_res_t = NaN
    solver.x_norm_res = NaN
    solver.y_norm_res = NaN
    solver.z_norm_res = NaN

    solver.primal_obj_t = NaN
    solver.dual_obj_t = NaN
    solver.primal_obj = NaN
    solver.dual_obj = NaN
    solver.gap = NaN
    solver.rel_gap = NaN
    solver.x_feas = NaN
    solver.y_feas = NaN
    solver.z_feas = NaN

    # preprocess and find initial point
    @timeit solver.timer "initialize" begin
        orig_model = solver.orig_model
        model = solver.model = Models.Model{T}(orig_model.c, orig_model.A, orig_model.b, orig_model.G, orig_model.h, orig_model.cones, obj_offset = orig_model.obj_offset) # copy original model to solver.model, which may be modified

        @timeit solver.timer "init_cone" point = solver.point = initialize_cone_point(solver.orig_model.cones, solver.orig_model.cone_idxs, solver.timer)

        if solver.reduce
            # TODO don't find point / unnecessary stuff before reduce
            @timeit solver.timer "remove_prim_eq" point.y = find_initial_y(solver, true)
            @timeit solver.timer "preproc_init_x" point.x = find_initial_x(solver)
        else
            @timeit solver.timer "preproc_init_x" point.x = find_initial_x(solver)
            @timeit solver.timer "preproc_init_y" point.y = find_initial_y(solver, false)
        end

        if solver.status != :SolveCalled
            point.x = fill(NaN, orig_model.n)
            point.y = fill(NaN, orig_model.p)
            return solver
        end

        solver.tau = one(T)
        solver.kap = one(T)
        calc_mu(solver)
        if isnan(solver.mu) || abs(one(T) - solver.mu) > sqrt(eps(T))
            @warn("initial mu is $(solver.mu) but should be 1 (this could indicate a problem with cone barrier oracles)")
        end
        Cones.load_point.(model.cones, point.primal_views)
        Cones.load_dual_point.(model.cones, point.dual_views)
    end

    # setup iteration helpers
    solver.x_residual = similar(model.c)
    solver.y_residual = similar(model.b)
    solver.z_residual = similar(model.h)

    solver.x_conv_tol = inv(max(one(T), norm(model.c)))
    solver.y_conv_tol = inv(max(one(T), norm(model.b)))
    solver.z_conv_tol = inv(max(one(T), norm(model.h)))
    solver.prev_is_slow = false
    solver.prev2_is_slow = false
    solver.prev_gap = NaN
    solver.prev_rel_gap = NaN
    solver.prev_x_feas = NaN
    solver.prev_y_feas = NaN
    solver.prev_z_feas = NaN

    stepper = solver.stepper
    @timeit solver.timer "setup_stepper" load(stepper, solver)
    @timeit solver.timer "setup_system" load(solver.system_solver, solver)

    # iterate from initial point
    while true
        @timeit solver.timer "calc_res" calc_residual(solver)

        @timeit solver.timer "calc_conv" calc_convergence_params(solver)

        @timeit solver.timer "print_iter" solver.verbose && print_iteration_stats(stepper, solver)

        @timeit solver.timer "check_conv" check_convergence(solver) && break

        if solver.num_iters == solver.iter_limit
            solver.verbose && println("iteration limit reached; terminating")
            solver.status = :IterationLimit
            break
        end
        if time() - start_time >= solver.time_limit
            solver.verbose && println("time limit reached; terminating")
            solver.status = :TimeLimit
            break
        end

        @timeit solver.timer "step" step(stepper, solver) || break
        solver.num_iters += 1


        # if solver.num_iters > 1
        #     error()
        # end
    end

    # calculate result and iteration statistics and finish
    point.x ./= solver.tau
    point.y ./= solver.tau
    point.z ./= solver.tau
    point.s ./= solver.tau
    Cones.load_point.(solver.model.cones, point.primal_views)
    Cones.load_dual_point.(model.cones, point.dual_views)

    solver.solve_time = time() - start_time

    # free memory used by some system solvers
    free_memory(solver.system_solver)

    solver.verbose && println("\nstatus is $(solver.status) after $(solver.num_iters) iterations and $(trunc(solver.solve_time, digits=3)) seconds\n")

    return solver
end

function calc_mu(solver::Solver{T}) where {T <: Real}
    solver.mu = (dot(solver.point.z, solver.point.s) + solver.tau * solver.kap) /
        (one(T) + solver.model.nu)
    return solver.mu
end

function calc_residual(solver::Solver{T}) where {T <: Real}
    model = solver.model
    point = solver.point

    # x_residual = -A'*y - G'*z - c*tau
    x_residual = solver.x_residual
    mul!(x_residual, model.G', point.z)
    mul!(x_residual, model.A', point.y, true, true)
    solver.x_norm_res_t = norm(x_residual)
    @. x_residual += model.c * solver.tau
    solver.x_norm_res = norm(x_residual) / solver.tau
    @. x_residual *= -1

    # y_residual = A*x - b*tau
    y_residual = solver.y_residual
    mul!(y_residual, model.A, point.x)
    solver.y_norm_res_t = norm(y_residual)
    @. y_residual -= model.b * solver.tau
    solver.y_norm_res = norm(y_residual) / solver.tau

    # z_residual = s + G*x - h*tau
    z_residual = solver.z_residual
    mul!(z_residual, model.G, point.x)
    @. z_residual += point.s
    solver.z_norm_res_t = norm(z_residual)
    @. z_residual -= model.h * solver.tau
    solver.z_norm_res = norm(z_residual) / solver.tau

    return
end

function calc_convergence_params(solver::Solver{T}) where {T <: Real}
    model = solver.model
    point = solver.point

    solver.prev_gap = solver.gap
    solver.prev_rel_gap = solver.rel_gap
    solver.prev_x_feas = solver.x_feas
    solver.prev_y_feas = solver.y_feas
    solver.prev_z_feas = solver.z_feas

    solver.primal_obj_t = dot(model.c, point.x)
    solver.dual_obj_t = -dot(model.b, point.y) - dot(model.h, point.z)
    solver.primal_obj = solver.primal_obj_t / solver.tau + model.obj_offset
    solver.dual_obj = solver.dual_obj_t / solver.tau + model.obj_offset
    solver.gap = dot(point.z, point.s)
    if solver.primal_obj < zero(T)
        solver.rel_gap = solver.gap / -solver.primal_obj
    elseif solver.dual_obj > zero(T)
        solver.rel_gap = solver.gap / solver.dual_obj
    else
        solver.rel_gap = NaN
    end

    solver.x_feas = solver.x_norm_res * solver.x_conv_tol
    solver.y_feas = solver.y_norm_res * solver.y_conv_tol
    solver.z_feas = solver.z_norm_res * solver.z_conv_tol

    return
end

function check_convergence(solver::Solver{T}) where {T <: Real}
    # check convergence criteria
    # TODO nearly primal or dual infeasible or nearly optimal cases?
    if max(solver.x_feas, solver.y_feas, solver.z_feas) <= solver.tol_feas &&
        (solver.gap <= solver.tol_abs_opt || (!isnan(solver.rel_gap) && solver.rel_gap <= solver.tol_rel_opt))
        solver.verbose && println("optimal solution found; terminating")
        solver.status = :Optimal
        return true
    end
    if solver.dual_obj_t > zero(T)
        infres_pr = solver.x_norm_res_t * solver.x_conv_tol / solver.dual_obj_t
        if infres_pr <= solver.tol_feas
            solver.verbose && println("primal infeasibility detected; terminating")
            solver.status = :PrimalInfeasible
            return true
        end
    end
    if solver.primal_obj_t < zero(T)
        infres_du = -max(solver.y_norm_res_t * solver.y_conv_tol, solver.z_norm_res_t * solver.z_conv_tol) / solver.primal_obj_t
        if infres_du <= solver.tol_feas
            solver.verbose && println("dual infeasibility detected; terminating")
            solver.status = :DualInfeasible
            return true
        end
    end
    if solver.mu <= solver.tol_feas * T(1e-2) && solver.tau <= solver.tol_feas * T(1e-2) * min(one(T), solver.kap)
        solver.verbose && println("ill-posedness detected; terminating")
        solver.status = :IllPosed
        return true
    end

    max_improve = zero(T)
    for (curr, prev) in ((solver.gap, solver.prev_gap), (solver.rel_gap, solver.prev_rel_gap),
        (solver.x_feas, solver.prev_x_feas), (solver.y_feas, solver.prev_y_feas), (solver.z_feas, solver.prev_z_feas))
        if isnan(prev) || isnan(curr)
            continue
        end
        max_improve = max(max_improve, (prev - curr) / (abs(prev) + eps(T)))
    end
    if max_improve < solver.tol_slow
        if solver.prev_is_slow && solver.prev2_is_slow
            solver.verbose && println("slow progress in consecutive iterations; terminating")
            solver.status = :SlowProgress
            return true
        else
            solver.prev2_is_slow = solver.prev_is_slow
            solver.prev_is_slow = true
        end
    else
        solver.prev2_is_slow = solver.prev_is_slow
        solver.prev_is_slow = false
    end

    return false
end

get_timer(solver::Solver) = solver.timer

get_status(solver::Solver) = solver.status
get_solve_time(solver::Solver) = solver.solve_time
get_num_iters(solver::Solver) = solver.num_iters

get_primal_obj(solver::Solver) = solver.primal_obj
get_dual_obj(solver::Solver) = solver.dual_obj

get_s(solver::Solver) = copy(solver.point.s)
get_z(solver::Solver) = copy(solver.point.z)

function get_x(solver::Solver{T}) where {T <: Real}
    if solver.preprocess && !iszero(solver.orig_model.n) && !any(isnan, solver.point.x)
        # unpreprocess solver's solution
        if solver.reduce && !iszero(solver.orig_model.p)
            # unreduce solver's solution
            # x0 = Q * [(R' \ b0), x]
            x = zeros(T, solver.orig_model.n - length(solver.reduce_Rpib0))
            x[solver.x_keep_idxs] = solver.point.x
            x = vcat(solver.reduce_Rpib0, x)
            lmul!(solver.reduce_Ap_Q, x)
            if !isempty(solver.reduce_row_piv_inv)
                x = x[solver.reduce_row_piv_inv]
            end
        else
            x = zeros(T, solver.orig_model.n)
            x[solver.x_keep_idxs] = solver.point.x
        end
    else
        x = copy(solver.point.x)
    end

    return x
end

function get_y(solver::Solver{T}) where {T <: Real}
    if solver.preprocess && !iszero(solver.orig_model.p) && !any(isnan, solver.point.y)
        # unpreprocess solver's solution
        y = zeros(T, solver.orig_model.p)
        if solver.reduce
            # unreduce solver's solution
            # y0 = R \ (-cQ1' - GQ1' * z0)
            y0 = solver.reduce_GQ1' * solver.point.z
            y0 .+= solver.reduce_cQ1
            @views ldiv!(solver.reduce_Ap_R, y0[1:length(solver.reduce_y_keep_idxs)])
            @. y[solver.reduce_y_keep_idxs] = -y0
        else
            y[solver.y_keep_idxs] = solver.point.y
        end
    else
        y = copy(solver.point.y)
    end

    return y
end

get_tau(solver::Solver) = solver.tau
get_kappa(solver::Solver) = solver.kap
get_mu(solver::Solver) = solver.mu

function load(solver::Solver{T}, model::Models.Model{T}) where {T <: Real}
    # @assert solver.status == :NotLoaded # TODO maybe want a reset function that just keeps options
    solver.orig_model = model
    solver.status = :Loaded
    return solver
end

include("initialize.jl")
include("stepper.jl")
include("systemsolvers/naive.jl")
include("systemsolvers/naiveelim.jl")
include("systemsolvers/symindef.jl")
include("systemsolvers/qrchol.jl")

# release memory used by sparse system solvers
free_memory(::SystemSolver) = nothing
free_memory(system_solver::Union{NaiveSparseSystemSolver, SymIndefSparseSystemSolver}) = free_memory(system_solver.fact_cache)

end
