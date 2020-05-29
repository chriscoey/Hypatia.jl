#=
Copyright 2019, Chris Coey and contributors

run examples tests from the examples folder and display basic benchmarks
=#

examples_dir = joinpath(@__DIR__, "../examples")
include(joinpath(examples_dir, "common.jl"))
using DataFrames
using Printf
using TimerOutputs

struct ExpInstances <: InstanceSet end
struct PolyhedralInstances <: InstanceSet end

run_in_bf = false

# options to solvers
timer = TimerOutput()
tol = 1e-8
default_solver_options = (
    verbose = true,
    iter_limit = 100,
    timer = timer,
    # system_solver = Solvers.NaiveDenseSystemSolver{Float64}(),
    system_solver = Solvers.QRCholDenseSystemSolver{Float64}(),
    # max_nbhd = 0.02,
    tol_rel_opt = tol,
    tol_abs_opt = tol,
    tol_feas = tol,
    )

# instance sets and real types to run and corresponding time limits (seconds)
instance_sets = [
    # (FastInstances, Float64, 60),
    (ExpInstances, Float64, 60),
    # (PolyhedralInstances, Float64, 60),
    ]

# types of models to run and corresponding options and example names
model_types = [
    # "native",
    "JuMP",
    ]

# list of names of native examples to run
native_example_names = [
    # "densityest",
    "linearopt",
    ]

# list of names of JuMP examples to run
JuMP_example_names = [
    # "centralpolymat",
    # "expdesign",
    # # "maxvolume", # TODO uncomment when geomean -> exp bridge is in
    # "portfolio",
    # "robustgeomprog",
    "signomialmin",
    ]

# start the tests

example_types = Tuple{String, Type{<:ExampleInstance}}[]
for mod_type in model_types, ex in eval(Symbol(mod_type, "_example_names"))
    ex_type = include(joinpath(examples_dir, ex, mod_type * ".jl"))
    push!(example_types, (ex, ex_type))
end

perf = DataFrame(
    example = Type{<:ExampleInstance}[],
    inst_set = Type{<:InstanceSet}[],
    real_T = Type{<:Real}[],
    count = Int[],
    inst_data = Tuple[],
    extender = Any[],
    test_time = Float64[],
    build_time = Float64[],
    status = Symbol[],
    solve_time = Float64[],
    iters = Int[],
    prim_obj = Float64[],
    dual_obj = Float64[],
    obj_diff = Float64[],
    compl = Float64[],
    x_viol = Float64[],
    y_viol = Float64[],
    z_viol = Float64[],
    ext_n = Int[],
    ext_p = Int[],
    ext_q = Int[],
    )

all_tests_time = time()

@testset "examples tests" begin
    for (ex_name, ex_type) in example_types, (inst_set, real_T, time_limit) in instance_sets
        ex_type_T = ex_type{real_T}
        instances = example_tests(ex_type_T, inst_set())
        isempty(instances) && continue
        println("\nstarting $(length(instances)) instances for $ex_type_T $inst_set\n")
        solver_options = (default_solver_options..., time_limit = time_limit)
        for (inst_num, inst) in enumerate(instances)
            if run_in_bf
                write_and_run(ex_type_T, inst...)
            else
                test_info = "$ex_type_T $inst_set $inst_num: $inst"
                @testset "$test_info" begin
                    println(test_info, "...")
                    test_time = @elapsed (extender, build_time, r) = test(ex_type_T, inst..., default_solver_options = solver_options)
                    push!(perf, (
                        ex_type, inst_set, real_T, inst_num, inst[1], extender, test_time, build_time,
                        r.status, r.solve_time, r.num_iters, r.primal_obj, r.dual_obj,
                        r.obj_diff, r.compl, r.x_viol, r.y_viol, r.z_viol,
                        r.n, r.p, r.q,
                        ))
                    @printf("... %8.2e seconds\n", test_time)
                end # write_only
            end
        end
    end

    # @printf("\nexamples tests total time: %8.2e seconds\n\n", time() - all_tests_time)
    # show(perf, allrows = true, allcols = true)
    # println("\n")
    # show(timer)
    # println("\n")
end

;
