#=
single_moi("strictmin_2D_43_dual", "try.csv", "hypatia", "qrchol_dense")
=#

import MathOptInterface
const MOI = MathOptInterface
const MOIB = MathOptInterface.Bridges
const MOIU = MathOptInterface.Utilities
const MOIF = MathOptInterface.FileFormats
using LinearAlgebra
import TimerOutputs
import MosekTools
const Mosek = MosekTools.Mosek
import SCS
import ECOS
import Hypatia
const SO = Hypatia.Solvers
import TimerOutputs

cblib_dir = joinpath(ENV["HOME"], "cblib/cblib.zib.de/download/all")
max_num_threads = length(Sys.cpu_info())
BLAS.set_num_threads(max_num_threads)
time_limit = 1800
iter_limit = 500
tol = 1e-8

system_solver_dict = Dict(
    "qrchol_dense" => SO.QRCholDenseSystemSolver{Float64}(),
    # "qrchol_sparse" => SO.QRCholSparseSystemSolver{Float64}(),
    "symindef_dense" => SO.SymIndefDenseSystemSolver{Float64}(),
    "naiveelim_dense" => SO.NaiveElimDenseSystemSolver{Float64}(),
    "symindef_sparse" => SO.SymIndefSparseSystemSolver{Float64}(),
    "naiveelim_sparse" => SO.NaiveElimSparseSystemSolver{Float64}(),
    "naive_sparse" => SO.NaiveSparseSystemSolver{Float64}(),
    "naive_dense" => SO.NaiveDenseSystemSolver{Float64}(),
    )

solver_dict = Dict(
    "mosek" => Mosek.Optimizer(
        QUIET = false,
        MSK_DPAR_INTPNT_CO_TOL_DFEAS = tol,
        MSK_DPAR_INTPNT_CO_TOL_PFEAS = tol,
        MSK_DPAR_INTPNT_CO_TOL_REL_GAP = tol,
        MSK_IPAR_NUM_THREADS = max_num_threads,
        MSK_IPAR_PRESOLVE_USE = 2, # (off, on, free)
        MSK_DPAR_OPTIMIZER_MAX_TIME = time_limit,
        ),
    "scs" => SCS.Optimizer(),
    "ecos" => ECOS.Optimizer(),
    "hypatia" => Hypatia.Optimizer{Float64}(
        # use_dense = false,
        verbose = true,
        system_solver = Hypatia.Solvers.QRCholDenseSystemSolver{Float64}(), # gets overwritten
        iter_limit = iter_limit,
        time_limit = time_limit,
        # tol_rel_opt = tol,
        # tol_abs_opt = tol,
        # tol_feas = tol,
        # tol_slow = 1e-7,
        )
    )

function get_result_attributes(solver_name, optimizer)
    step_t = ""
    step_b = ""
    if solver_name == "hypatia"
        hyp_solver = optimizer.optimizer.solver
        (abs_gap, rel_gap, x_feas, y_feas, z_feas) = (hyp_solver.gap, hyp_solver.rel_gap, hyp_solver.x_feas, hyp_solver.y_feas, hyp_solver.z_feas)
        status = hyp_solver.status
        num_iters = hyp_solver.num_iters
        if haskey(hyp_solver.timer, "step")
            step_t = TimerOutputs.time(hyp_solver.timer["step"]) / 1e9
            step_b = TimerOutputs.allocated(hyp_solver.timer["step"])
        end
        if haskey(hyp_solver.timer, "initialize")
            buildtime = TimerOutputs.time(hyp_solver.timer["initialize"]) / 1e9
        end
    else
        if solver_name == "mosek"
            num_iters = MOI.get(optimizer, MOI.BarrierIterations())
        else
            num_iters = ""
        end
        (abs_gap, rel_gap, x_feas, y_feas, z_feas) = ("", "", "", "", "")
        buildtime = ""
    end
    return (num_iters, buildtime, step_t, step_b, abs_gap, rel_gap, x_feas, y_feas, z_feas)
end

function bridge_optimizer(solver_name, optimizer)
    if solver_name == "mosek"
        return MOIB.Constraint.VectorSlack{Float64}(MOIB.Constraint.Scalarize{Float64}(optimizer))
    elseif solver_name == "ecos"
        # T = Float64;
        # bridged_model = MOIB.LazyBridgeOptimizer(optimizer);
        # MOIB.add_bridge(bridged_model, MOIB.Constraint.NonposToNonnegBridge{T});
        # MOIB.add_bridge(bridged_model, MOIB.Constraint.VectorizeBridge{T});
        # MOIB.add_bridge(bridged_model,  MOIB.Constraint.VectorFunctionizeBridge{T});
        bridged = MOIB.Constraint.NonposToNonneg{Float64}(optimizer);
        bridged = MOIB.Constraint.VectorFunctionize{Float64}(bridged);
        bridged = MOIB.Constraint.Vectorize{Float64}(bridged);
        bridged = MOIB.Constraint.RSOC{Float64}(bridged);
        return bridged
    elseif solver_name == "hypatia"
        return optimizer
    else
        error("haven't bridged $(solver_name) yet")
    end
end

function read_model(instname, precompiling)
    if !precompiling
        println("\nreading instance and constructing model...")
    end
    model = MOIF.CBF.Model()
    readtime = @elapsed MOI.read_from_file(model, joinpath(cblib_dir, instname * ".cbf.gz"))
    if !precompiling
        println("took $readtime seconds")
    end
    return model
end

function setup_optimizer(model, solver_name, system_solver_name, precompiling)
    if !precompiling
        println("building model...")
        flush(stdout)
    end
    setuptime = @elapsed begin
        cache = MOIU.UniversalFallback(MOIU.Model{Float64}());
        MOI.copy_to(cache, model);
        integer_indices = MOI.get(cache, MOI.ListOfConstraintIndices{MOI.SingleVariable, MOI.Integer}());
        MOI.delete.(cache, integer_indices);

        solver = solver_dict[solver_name]
        MOI.empty!(solver)
        if solver_name == "hypatia"
            solver.solver.system_solver = system_solver_dict[system_solver_name]
            if precompiling
                solver.solver.iter_limit = 3
            else
                solver.solver.iter_limit = iter_limit
            end
        end
        optimizer = MOIU.CachingOptimizer(MOIU.UniversalFallback(MOIU.Model{Float64}()), solver);
        bridged = bridge_optimizer(solver_name, optimizer);
        # bridged = MOIB.full_bridge_optimizer(optimizer, Float64);
        MOI.copy_to(bridged, cache);
    end
    if !precompiling
        println("took $setuptime seconds")
    end
    return (solver, optimizer, bridged)
end

function single_moi(instname, csvfile, solver_name, system_solver_name; print_timer = false, precompiling = false)
    model = read_model(instname, precompiling)
    (solver, optimizer, bridged) = setup_optimizer(model, solver_name, system_solver_name, precompiling)
    if !precompiling
        println("\nsolving model...")
        flush(stdout)
    end
    fdcsv = open(csvfile, "a")
    # try
        (val, moitime, bytes, gctime, memallocs) = (0.0, 0.0, 0.0, 0.0, 0.0)
        try
            (val, moitime, bytes, gctime, memallocs) = @timed MOI.optimize!(bridged)
        catch e
            println(e)
            flush(stdout)
        end

        if !precompiling
            println("took $moitime seconds")
            flush(stdout)
            status = MOI.get(bridged, MOI.TerminationStatus())
            if print_timer && solver_name == "hypatia" && status == MOI.OPTIMAL
                open(joinpath(dirname(csvfile), instname * system_solver_name * "_timer.txt"), "a") do tio
                    TimerOutputs.print_timer(tio, solver.solver.timer)
                    flush(tio)
                end
            end
            solvertime = MOI.get(bridged, MOI.SolveTime())
            primal_obj = MOI.get(bridged, MOI.ObjectiveValue())
            dual_obj = MOI.get(bridged, MOI.DualObjectiveValue())
            raw_status = MOI.get(bridged, MOI.RawStatusString())
            (num_iters, buildtime, step_t, step_b, abs_gap, rel_gap, x_feas, y_feas, z_feas) = get_result_attributes(solver_name, optimizer)
            print(fdcsv, "$status,$raw_status,$primal_obj,$dual_obj,$num_iters,$buildtime,$moitime,$gctime,$bytes,$solvertime,$step_t,$step_b," *
                "$abs_gap,$rel_gap,$x_feas,$y_feas,$z_feas"
                )
            flush(stdout)
        end
    # catch solveerror
    #     print(fdcsv, "SolverError,$(repeat(",", 16))")
    #     println("\n$(solver_name) errored: ", solveerror)
    # end
    close(fdcsv)
end

;
