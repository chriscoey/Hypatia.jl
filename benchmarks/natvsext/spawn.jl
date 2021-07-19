#=
utilities for spawning benchmark runs
=#

# reduce printing for worker
Base.eval(Distributed, :(function redirect_worker_output(ident, stream)
    @async while !eof(stream)
        println(readline(stream))
    end
end))

function spawn_step(
    fun::Function,
    fun_name::Symbol,
    time_limit::Real,
    worker::Int,
    )
    @assert nprocs() == 2
    status = :Success
    time_start = time()

    fut = Future()
    @async put!(fut, begin
        try
            remotecall_fetch(fun, worker)
        catch e
            println(fun_name, ": caught error: ", e)
        end
    end)

    killed_proc = false
    while !isready(fut)
        if Sys.free_memory() < free_memory_limit
            killstatus = :KilledMemory
        elseif time() - time_start > time_limit
            killstatus = :KilledTime
        else
            sleep(1)
            continue
        end
        try
            if !isready(fut)
                interrupt()
                sleep(5)
            end
            if !isready(fut) && nprocs() > 1
                w = workers()[end]
                run(`kill -SIGKILL $(remotecall_fetch(getpid, w))`)
                killed_proc = true
                sleep(10)
            end
        catch e
        end
        status = Symbol(fun_name, killstatus)
        println("status: ", status)
        break
    end

    output = (killed_proc || !isready(fut)) ? nothing : fetch(fut)
    if isnothing(output)
        if status == :Success
            status = Symbol(fun_name, :CaughtError)
        end
        output = NamedTuple()
    end
    finalize(fut)
    flush(stdout); flush(stderr)

    return (status, output)
end

function spawn_instance(
    ex_name::String,
    ex_type::Type{<:Examples.ExampleInstanceJuMP},
    compile_inst::Tuple,
    inst_data::Tuple,
    extender::Union{Symbol, Nothing},
    solver::Tuple,
    solve::Bool,
    num_threads::Int,
    )
    worker = addprocs(1, enable_threaded_blas = true,
        exeflags = `--threads $num_threads`)[1]
    @assert nprocs() == 2
    println("loading files")
    @fetchfrom worker begin
        @eval import LinearAlgebra
        LinearAlgebra.BLAS.set_num_threads(num_threads)
        @eval using MosekTools
        @eval using ECOS
        include(joinpath(@__DIR__, "../../examples/Examples.jl"))
        @eval using Main.Examples
        flush(stdout); flush(stderr)
        return
    end
    println("running compile instance")
    original_stdout = stdout
    (out_rd, out_wr) = redirect_stdout() # don't print output
    @fetchfrom worker begin
        Examples.run_instance(ex_type, compile_inst, extender, NamedTuple(), solver[2],
            default_options = solver[3], test = false)
        flush(stdout); flush(stderr)
        return
    end
    redirect_stdout(original_stdout)
    close(out_wr)
    println("finished compile instance")

    println("\nsetup model")
    print_memory()

    setup_model_args = (ex_type, inst_data, extender, solver[3], solver[2])
    setup_fun() = @eval begin
        (model, model_stats) = Examples.setup_model($setup_model_args...)
        GC.gc()
        return model_stats
    end
    setup_time = @elapsed (script_status, model_stats) =
        spawn_step(setup_fun, :SetupModel, setup_time_limit, worker)
    setup_killed = (script_status != :Success)
    if setup_killed
        println("setup model failed: $script_status")
        model_stats = NamedTuple()
    end

    if solve && !setup_killed
        println("\nsolve and check")
        print_memory()
        check_fun() = @eval begin
            solve_stats = Examples.solve_check(model, test = false)
            return solve_stats
        end
        check_time = @elapsed (script_status, solve_stats) =
            spawn_step(check_fun, :SolveCheck, check_time_limit, worker)
        check_killed = (script_status != :Success)
        check_killed && println("solve and check failed: $script_status")
    else
        solve_stats = NamedTuple()
        check_time = 0.0
        check_killed = true
    end
    if check_killed
        if script_status == :Success
            @assert !solve
            script_status = :SkippedSolveCheck
        end
        solver_hit_limit = true
    else
        solver_hit_limit = (string(solve_stats.status) == "TimeLimit")
        solver_hit_limit && println("solver hit time limit")
    end

    try
        wait(rmprocs(worker))
    catch e
        @warn("error during process shutdown: ", e)
    end
    @assert nprocs() == 1

    script_status = string(script_status)
    run_perf = (; model_stats..., solve_stats..., setup_time,
        check_time, script_status)
    return (setup_killed, solver_hit_limit, run_perf)
end
