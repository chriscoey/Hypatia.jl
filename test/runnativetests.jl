#=
Copyright 2019, Chris Coey and contributors

run native instance tests from test/nativeinstances.jl and display basic benchmarks
=#

using Test
using DataFrames
using Printf
using TimerOutputs
import Hypatia
import Hypatia.Solvers

include(joinpath(@__DIR__, "nativeinstances.jl"))
include(joinpath(@__DIR__, "nativesets.jl"))

all_reals = [
    Float64,
    # Float32,
    # BigFloat,
    ]
default_reals = [
    Float64,
    ]

# system solvers tests options
system_solvers_instance_names = vcat(
    # inst_preproc,
    # inst_infeas,
    # inst_cones_few, # NOTE subset of inst_cones_many
    inst_cones_many,
    )
system_solvers = Dict(
    "NaiveDense" => all_reals,
    # "NaiveSparse" => default_reals,
    # # "NaiveIndirect" => all_reals, # TODO fix
    # "NaiveElimDense" => all_reals,
    # "NaiveElimSparse" => default_reals,
    # "SymIndefDense" => all_reals,
    # "SymIndefSparse" => default_reals,
    # "QRCholDense" => all_reals,
    )

# preprocessing test options
preprocess_instance_names = vcat(
    # inst_preproc,
    # inst_infeas,
    # inst_cones_few, # NOTE subset of inst_cones_many
    # inst_cones_many,
    )
preprocess_system_solver = "SymIndefDense"
preprocess_reals = all_reals
preprocess_options = (init_use_indirect = false, reduce = false)
preprocess_flags = [
    true,
    false,
    ]

# indirect initialization test options
init_use_indirect_instance_names = vcat(
    # inst_preproc,
    # inst_infeas,
    # inst_cones_few, # NOTE subset of inst_cones_many
    # inst_cones_many,
    )
init_use_indirect_system_solver = "SymIndefDense"
init_use_indirect_reals = all_reals
init_use_indirect_options = (preprocess = false, reduce = false)
init_use_indirect_flags = [
    true,
    false,
    ]

# reduce test options
reduce_instance_names = vcat(
    # inst_preproc,
    # inst_infeas,
    # inst_cones_few, # NOTE subset of inst_cones_many
    # inst_cones_many,
    )
reduce_system_solver = "QRCholDense"
reduce_reals = all_reals
reduce_options = (preprocess = true, init_use_indirect = false)
reduce_flags = [
    true,
    false,
    ]

# other solver options
timer = TimerOutput()
other_options = (
    verbose = true,
    # verbose = false,
    iter_limit = 250,
    time_limit = 6e1,
    timer = timer,
    )

@info("starting native tests")

perf = DataFrame(
    inst_name = String[],
    sys_solver = String[],
    real_T = String[],
    preprocess = Bool[],
    init_use_indirect = Bool[],
    reduce = Bool[],
    test_time = Float64[],
    )

function run_instance_options(T::Type{<:Real}, inst_name::String, sys_name::String, test_info::String; kwargs...)
    @testset "$test_info" begin
        println(test_info, "...")
        inst_function = eval(Symbol(inst_name))
        sys_solver = Solvers.eval(Symbol(sys_name, "SystemSolver"))
        solver = Solvers.Solver{T}(; system_solver = sys_solver{T}(), kwargs..., other_options...)
        test_time = @elapsed inst_function(T, solver = solver)
        push!(perf, (inst_name, sys_name, string(T), solver.preprocess, solver.init_use_indirect, solver.reduce, test_time))
        @printf("... %8.2e seconds\n", test_time)
    end
    return nothing
end

all_tests_time = time()

@testset "native tests" begin
    for inst_name in system_solvers_instance_names, (sys_name, real_types) in system_solvers, T in real_types
        run_instance_options(T, inst_name, sys_name, "$inst_name system_solver = $sys_name $T")
    end

    for inst_name in preprocess_instance_names, preprocess in preprocess_flags, T in preprocess_reals
        run_instance_options(T, inst_name, preprocess_system_solver, "$inst_name preprocess = $preprocess $T"; preprocess = preprocess, preprocess_options...)
    end

    for inst_name in init_use_indirect_instance_names, init_use_indirect in init_use_indirect_flags, T in init_use_indirect_reals
        run_instance_options(T, inst_name, init_use_indirect_system_solver, "$inst_name init_use_indirect = $init_use_indirect $T"; init_use_indirect = init_use_indirect, init_use_indirect_options...)
    end

    for inst_name in reduce_instance_names, reduce in reduce_flags, T in reduce_reals
        run_instance_options(T, inst_name, reduce_system_solver, "$inst_name reduce = $reduce $T"; reduce = reduce, reduce_options...)
    end

    # @printf("\nnative tests total time: %8.2e seconds\n\n", time() - all_tests_time)
    # show(perf, allrows = true, allcols = true)
    # println("\n")
    # show(timer)
    # println("\n")
end
;
