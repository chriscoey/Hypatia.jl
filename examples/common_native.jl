#=
common code for native examples
=#

include(joinpath(@__DIR__, "common.jl"))

abstract type ExampleInstanceNative{T <: Real} <: ExampleInstance{T} end

# fallback: just check optimal status
function test_extra(inst::ExampleInstanceNative, result::NamedTuple)
    @test result.status == Solvers.Optimal
end

function run_instance(
    ex_type::Type{<:ExampleInstanceNative{T}}, # an instance of a native example
    inst_data::Tuple,
    inst_options::NamedTuple = NamedTuple(),
    solver_type::Type{<:Solvers.Solver} = Solvers.Solver{T}; # TODO can generalize for other solvers
    default_options::NamedTuple = NamedTuple(),
    test::Bool = true,
    rseed::Int = 1,
    ) where {T <: Real}
    new_options = merge(default_options, inst_options)

    println("setup model")
    setup_time = @elapsed begin
        Random.seed!(rseed)
        inst = ex_type(inst_data...)
        model = build(inst)

        string_cones = [string(nameof(c)) for c in unique(typeof.(model.cones))]
        model_stats = (model.n, model.p, model.q, string_cones)

        solver = Solvers.Solver{T}(; default_options..., inst_options...)
    end
    flush(stdout); flush(stderr)

    println("solve and check")
    check_time = @elapsed begin
        Solvers.load(solver, model)
        Solvers.solve(solver)
        flush(stdout); flush(stderr)

        solve_stats = process_result(model, solver)

        if test
            named_result = NamedTuple{(:status, :solve_time, :num_iters, :primal_obj, :dual_obj, :obj_diff, :compl, :x_viol, :y_viol, :z_viol, :x, :y, :z, :s)}(solve_stats)
            test_extra(inst, named_result)
        end
    end
    flush(stdout); flush(stderr)

    return (model_stats..., string(solve_stats[1]), solve_stats[2:(end - 4)]..., setup_time, check_time)
end
