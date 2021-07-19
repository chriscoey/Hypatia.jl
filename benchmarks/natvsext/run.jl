#=
run natvsext benchmarks
see natvsext/README.md
=#

using Test
using Printf
import DataFrames
import CSV
import Hypatia
using MosekTools
using ECOS

include(joinpath(@__DIR__, "../../examples/Examples.jl"))
using Main.Examples

using Distributed
include(joinpath(@__DIR__, "spawn.jl"))

# path to write results DataFrame to CSV
results_path = joinpath(mkpath(joinpath(@__DIR__, "raw")), "ecos.csv")

# option to keep setting up larger models, only if solver is Hypatia,
# even if last solve was killed
# setup_model_anyway = true
setup_model_anyway = false

verbose = true # make solvers print output
# verbose = false
num_threads = 16 # number of threads for BLAS and Julia processes running instances
free_memory_limit = 8 * 2^30 # keep at least X GB of RAM available
optimizer_time_limit = 1800
setup_time_limit = 1.2 * optimizer_time_limit
check_time_limit = 1.2 * optimizer_time_limit
tol_loose = 1e-7
tol_tight = 1e-3 * tol_loose

hyp_solver = ("Hypatia", Hypatia.Optimizer, (
    verbose = verbose,
    iter_limit = 250,
    time_limit = optimizer_time_limit,
    tol_abs_opt = tol_tight,
    tol_rel_opt = tol_loose,
    tol_feas = tol_loose,
    tol_infeas = tol_tight,
    init_use_indirect = true, # skips dual equalities preprocessing
    use_dense_model = true,
    ))

mosek_solver = ("Mosek", Mosek.Optimizer, (
    QUIET = !verbose,
    MSK_IPAR_NUM_THREADS = num_threads,
    MSK_IPAR_OPTIMIZER = Mosek.MSK_OPTIMIZER_CONIC,
    MSK_IPAR_INTPNT_BASIS = Mosek.MSK_BI_NEVER, # no basis identification for LP
    MSK_DPAR_OPTIMIZER_MAX_TIME = optimizer_time_limit,
    MSK_DPAR_INTPNT_CO_TOL_REL_GAP = tol_loose,
    MSK_DPAR_INTPNT_CO_TOL_PFEAS = tol_loose,
    MSK_DPAR_INTPNT_CO_TOL_DFEAS = tol_loose,
    MSK_DPAR_INTPNT_CO_TOL_INFEAS = tol_tight,
    ))

ecos_solver = ("Ecos", ECOS.Optimizer, (
    verbose = 3 * verbose,
    reltol = tol_loose,
    abstol = tol_tight,
    feastol = tol_tight,
    ))

# instance sets and solvers to run
inst_sets = [
    #= natural formulations paper =#
    # ("nat", hyp_solver),
    # ("ext", hyp_solver),
    # ("ext", mosek_solver),
    # ("extEP", hyp_solver), # ExpPSD extender
    # ("extSEP", hyp_solver), # SOCExpPSD extender
    # ("extEP", mosek_solver), # ExpPSD extender
    # ("extSEP", mosek_solver), # SOCExpPSD extender
    #= spectral function cones paper =#
    ("nat", hyp_solver),
    ("ext", hyp_solver),
    ("ext", mosek_solver),
    ("ext", ecos_solver),
    ("logdet", hyp_solver),
    ("sepspec", hyp_solver),
    ("direct", hyp_solver),
    ("direct", mosek_solver),
    ("eigord", hyp_solver),
    ("eigord", mosek_solver),
    ]

# models to run
JuMP_examples = [
    #= natural formulations paper =#
    # "densityest",
    # "doptimaldesign",
    # "matrixcompletion",
    # "matrixregression",
    # "polymin",
    # "portfolio",
    # "shapeconregr",
    #= WSOS cones paper =#
    # "nearestpolymat",
    # "polynorm",
    #= spectral function cones paper =#
    # "centralpolymat",
    # "classicalquantum",
    # "covarianceest",
    # "experimentdesign",
    "nonparametricdistr",
    ]

interrupt()
@assert nprocs() == 1
println()

print_memory() = println("free memory (GB): ", Float64(Sys.free_memory()) / 2^30)
print_memory()

extender_name(ext::Nothing) = missing
extender_name(ext::Symbol) = string(ext)

println("\nstarting benchmark runs")
time_all = time()

@testset "examples tests" begin
perf = Examples.setup_benchmark_dataframe()
CSV.write(results_path, perf)

@testset "$ex" for ex in JuMP_examples
(ex_type, ex_insts) = Examples.get_benchmark_instances("JuMP", ex)
@testset "$inst_set, $(solver[1])" for (inst_set, solver) in inst_sets
    haskey(ex_insts, inst_set) || continue
    (extender, inst_subsets) = ex_insts[inst_set]
    isempty(inst_subsets) && continue

    info_perf = (; inst_set, :example => ex, :model_type => "JuMP",
        :real_T => Float64, :solver => solver[1],
        :extender => extender_name(extender))
    str = "$ex $inst_set $(solver[1])"
    println("\nstarting $str")

    for inst_subset in inst_subsets
        solve = true
        # first instance is only used for compilation
        compile_inst = inst_subset[1]
        for (inst_num, inst_data) in enumerate(inst_subset[2:end])
            println("\nstarting $str $inst_num $inst_data")
            flush(stdout); flush(stderr)

            total_time = @elapsed (setup_killed, check_killed, run_perf) =
                spawn_instance(ex, ex_type{Float64}, compile_inst,
                inst_data, extender, solver, solve, num_threads)

            new_perf = (; info_perf..., run_perf..., total_time,
                inst_num, inst_data)
            Examples.write_perf(perf, results_path, new_perf)
            @printf("%8.2e seconds\n", total_time)
            flush(stdout); flush(stderr)

            setup_killed && break
            if check_killed
                if setup_model_anyway && (solver[1] == "Hypatia")
                    solve = false
                else
                    break
                end
            end
        end
    end
end
end

flush(stdout); flush(stderr)
println("\n")
DataFrames.show(perf, allrows = true, allcols = true)
println("\n")
flush(stdout); flush(stderr)
end

interrupt()
@printf("\nbenchmarks total time: %8.2e seconds\n\n", time() - time_all)
;
