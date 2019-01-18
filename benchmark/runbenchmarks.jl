#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

TODO readme for benchmarks and describe ARGS for running on command line
=#

using Pkg; Pkg.activate("..") # TODO delete later

using Hypatia
using MathOptFormat
using MathOptInterface
MOI = MathOptInterface
using GZip
using Dates

function read_into_model(filename::String)
    if endswith(filename, ".gz")
        io = GZip.open(filename, "r")
    else
        io = open(filename, "r")
    end
    if endswith(filename, ".cbf.gz") || endswith(filename, ".cbf")
        model = MathOptFormat.CBF.Model()
    elseif endswith(filename, ".mof.json.gz") || endswith(filename, ".mof.json")
        model = MathOptFormat.MOF.Model()
    else
        error("MathOptInterface.read_from_file is not implemented for this filetype: $filename")
    end
    MOI.read_from_file(model, io)
    return model
end

# parse command line arguments
println()
if length(ARGS) != 3
    error("usage: julia runbenchmarks.jl instance_set input_path output_path")
end

instanceset = ARGS[1]
instsetfile = joinpath(@__DIR__, "instancesets", instanceset)
if !isfile(instsetfile)
    error("instance set file not found: $instsetfile")
end

inputpath = ARGS[2]
if !isdir(inputpath)
    error("input path is not a valid directory: $inputpath")
end

# check that each instance is in the inputpath
instances = SubString[]
for l in readlines(instsetfile)
    str = split(strip(l))
    if !isempty(str)
        str1 = first(str)
        if !startswith(str1, '#')
            push!(instances, str1)
        end
    end
end
println("instance set $instanceset contains $(length(instances)) instances")
for instname in instances
    instfile = joinpath(inputpath, instname)
    if !isfile(instfile)
        error("instance file not found: $instfile")
    end
end

outputpath = ARGS[3]
if !isdir(outputpath)
    error("output path is not a valid directory: $outputpath")
end

# Hypatia options
verbose = true
timelimit = 1e2
lscachetype = Hypatia.QRSymmCache
usedense = false

MOI.Utilities.@model(HypatiaModelData,
    (MOI.Integer,), # integer constraints will be ignored by Hypatia
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
        MOI.SecondOrderCone, MOI.RotatedSecondOrderCone,
        MOI.PositiveSemidefiniteConeTriangle,
        MOI.ExponentialCone),
    (MOI.PowerCone,),
    (MOI.SingleVariable,),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    )

optimizer = MOI.Utilities.CachingOptimizer(HypatiaModelData{Float64}(), Hypatia.Optimizer(
    verbose = verbose,
    timelimit = timelimit,
    lscachetype = lscachetype,
    usedense = usedense,
    tolrelopt = 1e-6,
    tolabsopt = 1e-7,
    tolfeas = 1e-7,
    ))

println("\nstarting benchmark run in 5 seconds\n")
sleep(5.0)

# each line of csv file will summarize Hypatia performance on a particular instance
csvfile = joinpath(outputpath, "RESULTS_$(instanceset).csv")
open(csvfile, "w") do fdcsv
    println(fdcsv, "instname,status,pobj,dobj,niters,runtime,gctime,bytes")
end

# run each instance, print Hypatia output to instance-specific file, and print results to a single csv file
OUT = stdout
ERR = stderr
for instname in instances
    println("starting $instname")

    solveerror = nothing
    (status, pobj, dobj, niters, runtime, gctime, bytes) = (:UnSolved, NaN, NaN, -1, NaN, NaN, -1)
    memallocs = nothing

    instfile = joinpath(outputpath, instname * ".txt")
    open(instfile, "w") do fdinst
        redirect_stdout(fdinst)
        redirect_stderr(fdinst)

        println("instance $instname")
        println("ran at: ", Dates.now())
        println()

        println("\nreading instance and constructing model...")
        readtime = @elapsed begin
            model = read_into_model(joinpath(inputpath, instname))
            MOI.empty!(optimizer)
            MOI.copy_to(optimizer, model)
        end
        println("took $readtime seconds")

        println("\nsolving model...")
        try
            (val, runtime, bytes, gctime, memallocs) = @timed MOI.optimize!(optimizer)
            println("\nHypatia finished")
            status = MOI.get(optimizer, MOI.TerminationStatus())
            niters = -1 # TODO niters = MOI.get(optimizer, MOI.BarrierIterations())
            pobj = MOI.get(optimizer, MOI.ObjectiveValue())
            dobj = MOI.get(optimizer, MOI.ObjectiveBound())
        catch solveerror
            println("\nHypatia errored: ", solveerror)
        end
        println("took $runtime seconds")
        println("memory allocation data:")
        dump(memallocs)
        println()

        redirect_stdout(OUT)
        redirect_stderr(ERR)
    end

    if !isnothing(solveerror)
        println("Hypatia errored: ", solveerror)
    end

    open(csvfile, "a") do fdcsv
        println(fdcsv, "$instname,$status,$pobj,$dobj,$niters,$runtime,$gctime,$bytes")
    end
end

println("\ndone\n")
