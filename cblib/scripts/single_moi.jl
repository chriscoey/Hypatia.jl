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
import Hypatia
const SO = Hypatia.Solvers

cblib_dir = joinpath(ENV["HOME"], "cblib/cblib.zib.de/download/all")
max_num_threads = length(Sys.cpu_info())
BLAS.set_num_threads(max_num_threads)
time_limit = 1800
iter_limit = 500
tol = sqrt(eps())
system_solver_dict = Dict(
    "qrchol_dense" => SO.QRCholDenseSystemSolver,
    # "qrchol_sparse" => SO.QRCholSparseSystemSolver,
    "symindef_dense" => SO.SymIndefDenseSystemSolver,
    "naiveelim_dense" => SO.NaiveElimDenseSystemSolver,
    "symindef_sparse" => SO.SymIndefSparseSystemSolver,
    "naiveelim_sparse" => SO.NaiveElimSparseSystemSolver,
    "naive_sparse" => SO.NaiveSparseSystemSolver,
    "naive_dense" => SO.NaiveDenseSystemSolver,
    )
options = (
    verbose = true,
    iter_limit = iter_limit,
    time_limit = time_limit,
    tol_rel_opt = tol,
    tol_abs_opt = tol,
    tol_feas = tol,
    )
# couldn't get general version to work
convert_cone(cone::Hypatia.Cones.Nonnegative, out_type::Type) = Hypatia.Cones.Nonnegative{out_type}(cone.dim)
convert_cone(cone::Hypatia.Cones.HypoPerLog, out_type::Type) = Hypatia.Cones.HypoPerLog{out_type}(cone.dim)

function get_result_attributes(optimizer)
    step_t = ""
    step_b = ""
    hyp_solver = optimizer.solver
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
    return (num_iters, buildtime, step_t, step_b, abs_gap, rel_gap, x_feas, y_feas, z_feas)
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

function setup_optimizer(model, precompiling)
    if !precompiling
        println("building model...")
        flush(stdout)
    end
    setuptime = @elapsed begin
        cache = MOIU.UniversalFallback(MOIU.Model{Float64}());
        MOI.copy_to(cache, model);
        integer_indices = MOI.get(cache, MOI.ListOfConstraintIndices{MOI.SingleVariable, MOI.Integer}());
        MOI.delete.(cache, integer_indices);

        optimizer = Hypatia.Optimizer()
        MOI.empty!(optimizer)
        if precompiling
            optimizer.solver.iter_limit = 3
        else
            optimizer.solver.iter_limit = iter_limit
        end
        MOI.copy_to(optimizer, cache);
    end
    if !precompiling
        println("took $setuptime seconds")
    end
    return optimizer
end

function single_moi(
    instname,
    csvfile,
    system_solver_name;
    print_timer = false,
    precompiling = false,
    out_type = Float64,
    )
    model = read_model(instname, precompiling)
    mock_optimizer = setup_optimizer(model, precompiling)

    model = mock_optimizer.model
    new_model = Hypatia.Models.Model{out_type}(
        out_type.(model.c),
        out_type.(model.A),
        out_type.(model.b),
        out_type.(model.G),
        out_type.(model.h),
        convert_cone.(model.cones, out_type),
        )
    solver = SO.Solver{out_type}(; options..., system_solver = system_solver_dict[system_solver_name]{out_type}())
    SO.load(solver, new_model)
    # TODO there is no need to attach solver to an Optimizer
    optimizer = Hypatia.Optimizer{out_type}()
    optimizer.model = new_model
    optimizer.solver = solver

    if !precompiling
        println("\nsolving model...")
        flush(stdout)
    end
    fdcsv = open(csvfile, "a")
    # try
        (val, moitime, bytes, gctime, memallocs) = (0.0, 0.0, 0.0, 0.0, 0.0)
        try
            (val, moitime, bytes, gctime, memallocs) = @timed SO.solve(solver)
        catch e
            println(e)
            flush(stdout)
        end

        if !precompiling
            println("took $moitime seconds")
            flush(stdout)
            status = MOI.get(optimizer, MOI.TerminationStatus())
            if print_timer && status == MOI.OPTIMAL
                open(joinpath(dirname(csvfile), instname * system_solver_name * "_timer.txt"), "a") do tio
                    TimerOutputs.print_timer(tio, optimizer.solver.timer)
                    flush(tio)
                end
            end
            solvertime = MOI.get(optimizer, MOI.SolveTime())
            primal_obj = MOI.get(optimizer, MOI.ObjectiveValue())
            dual_obj = MOI.get(optimizer, MOI.DualObjectiveValue())
            raw_status = MOI.get(optimizer, MOI.RawStatusString())
            (num_iters, buildtime, step_t, step_b, abs_gap, rel_gap, x_feas, y_feas, z_feas) = get_result_attributes(optimizer)
            print(fdcsv, "$status,$raw_status,$primal_obj,$dual_obj,$num_iters,$buildtime,$moitime,$gctime,$bytes,$solvertime,$step_t,$step_b," *
                "$abs_gap,$rel_gap,$x_feas,$y_feas,$z_feas,normalF64"
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
