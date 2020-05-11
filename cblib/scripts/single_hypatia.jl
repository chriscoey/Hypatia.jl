#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

this file is to be deprecated, previously used to run hypatia separately to
other moi solvers, is lighter to precompile and reads from jlds as opposed to
cbf, for problems outside of cblib

=#

# using Revise
import Hypatia
const CO = Hypatia.Cones
const MO = Hypatia.Models
const SO = Hypatia.Solvers
import JLD
import TimerOutputs
using LinearAlgebra
using SparseArrays

max_num_threads = length(Sys.cpu_info())
BLAS.set_num_threads(max_num_threads)

solvermap = Dict(
    "qrchol" => SO.QRCholSystemSolver{Float64}(use_sparse = false),
    "symindef" => SO.SymIndefSystemSolver{Float64}(use_sparse = true),
    "naiveelim" => SO.NaiveElimSystemSolver{Float64}(use_sparse = true),
    )

# Hypatia options
# TODO copy to run_single so run_moi can reuse
T = Float64
verbose = true
time_limit = 1800
use_infty_nbhd = true
max_nbhd = T(0.7)
tol = 1e-8
iter_limit = 500

jld_dir = joinpath(@__DIR__(), "../instances")

function single_hypatia(instname, csvfile, system_solver_name)

    println("\nreading instance and constructing model...")
    readtime = @elapsed begin
        md = JLD.load(joinpath(jld_dir, instname * ".jld"))
        (c, A, b, G, h, cones) = (md["c"], md["A"], md["b"], md["G"], md["h"], md["cones"])
        flush(stdout)
    end
    println("took $readtime seconds")

    buildtime = @elapsed begin
        model = MO.Model{T}(c, A, b, G, h, cones)
        solver = SO.Solver{T}(system_solver = solvermap[system_solver_name], use_infty_nbhd = use_infty_nbhd, max_nbhd = max_nbhd,
            tol_abs_opt = tol, tol_rel_opt = tol, tol_feas = tol, time_limit = time_limit, iter_limit = iter_limit, verbose = verbose)
        SO.load(solver, model)
    end

    println("\nsolving model...")
    fdcsv = open(csvfile, "a")
    try
        (val, runtime, bytes, gctime, memallocs) = @timed SO.solve(solver)
        flush(stdout)
        (abs_gap, rel_gap, x_feas, y_feas, z_feas) = (solver.gap, solver.rel_gap, solver.x_feas, solver.y_feas, solver.z_feas)
        flush(stdout)
        println("\nHypatia finished")
        status = solver.status
        niters = solver.num_iters
        primal_obj = solver.primal_obj
        dual_obj = solver.dual_obj
        flush(stdout)
        directions_t = TimerOutputs.time(solver.timer["step"]["directions"]) / 1e9
        directions_b = TimerOutputs.allocated(solver.timer["step"]["directions"])
        open(joinpath(dirname(csvfile), instname * system_solver_name * "_timer.txt"), "a") do tio
            TimerOutputs.print_timer(tio, solver.timer)
        end
        print(fdcsv, "$status,$status,$primal_obj,$dual_obj,$niters,$readtime,$buildtime,$runtime,$gctime,$bytes,$runtime,$directions_t,$directions_b," *
            "$abs_gap,$rel_gap,$x_feas,$y_feas,$z_feas"
            )
        flush(fdcsv)
        println("took $runtime seconds")
        println("memory allocation data:")
        dump(memallocs)
    catch solveerror
        print(fdcsv, "SolverError,,,,,,,,,,,,,,,")
        println("\nHypatia errored: ", solveerror)
    end
    close(fdcsv)
    return
end
