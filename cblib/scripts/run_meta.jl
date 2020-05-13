#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

j scripts/run_meta.jl sample sample 1800 30000000
j cblib/scripts/run_meta.jl exporthantsmall exporthantsmall 900 30000000
j -J sc_img.so scripts/run_meta.jl cbf_easy easy 1800 30000000
julia scripts/run_meta.jl cbf_large linsystems 15 99999999
=#

using CSV

include(joinpath(@__DIR__, "read_instances.jl"))
# need this in case we do a compile run
# include(joinpath(@__DIR__, "single_hypatia.jl"))

println()

if length(ARGS) == 4
    set = ARGS[1]
    hostname = chomp(read(`hostname`, String))
    outsubfolder = ARGS[2]
    outputpath = joinpath(outsubfolder, hostname)
    tlim = parse(Float64, ARGS[3])
    mlim = parse(Float64, ARGS[4])
else
    error("usage: julia run_meta.jl instance_set output_path tlim mlim")
end

setfile = joinpath(@__DIR__, "../sets", set * ".txt")
if !isfile(setfile)
    error("instance set file not found: $setfile")
end
!isdir(outsubfolder) && mkdir(outsubfolder)
!isdir(outputpath) && mkdir(outputpath)

cblib_dir = joinpath(ENV["HOME"], "cblib/cblib.zib.de/download/all")

fdmeta = open(joinpath(outputpath, "META.txt"), "w")
println(fdmeta, "#INSTANCES#")

# check that each instance is in the inputpath
instances = read_instances(setfile)
println(fdmeta, "instance set $set contains $(length(instances)) instances")
for instname in instances
    instfile = joinpath(cblib_dir, instname * ".cbf.gz")
    println(fdmeta, instname)
    if !isfile(instfile)
        error("instance file not found: $instfile")
    end
end

println(fdmeta)
flush(fdmeta)

system_solvers = [
    # "qrchol_sparse",
    # "symindef_sparse",
    # "naiveelim_sparse",
    # "naive_sparse",
    # "naive_dense",
    "qrchol_dense",
    ]

moi_solvers = [
    "hypatia",
    # "mosek",
    # "ecos",
    ]

if length(moi_solvers) > 1 && length(system_solvers) > 1
    error("cannot run MOI solvers with Hypatia's system solvers")
end

println("\nstarting benchmark run in 1 seconds\n")
sleep(1.0)

# each line of csv file will summarize performance on a particular instance
csvfile = joinpath(outputpath, "RESULTS_$(set).csv")
open(csvfile, "a") do fdcsv
    print(fdcsv, "\ninstname,solver,systemsolver,status,rawstatus,primalobj,dualobj,niters,buildtime,moitime,gctime,bytes," *
        "solvertime,directionstime,directionsbytes,absgap,relgap,xfeas,yfeas,zfeas,notes")
    flush(fdcsv)
end
flush(fdmeta)

for instname in instances, solver in moi_solvers, ss in system_solvers
    # system_solvers = (solver == "hypatia" ? hypatia_system_solvers : [""])

    stats = CSV.read(joinpath(@__DIR__(), "../stats/STATS_all.csv"))
    if solver == "ecos"
        if stats[stats.instname .== instname, :semidefinitereal][1] > 0 ||
            stats[stats.instname .== instname, :power][1] > 0
            println(fdmeta, "\nskipping $instname due to cone types")
            flush(fdmeta)
            continue
        end
    end

    # if stats[stats.instname .== instname, :semidefinitereal][1] > 0 ||
    #     stats[stats.instname .== instname, :power][1] > 0 ||
    #     stats[stats.instname .== instname, :hypoperlog][1] > 0
    #     println(fdmeta, "\nskipping $instname due to cone types")
    #     flush(fdmeta)
    #     continue
    # end

    filesize = stats[stats.instname .== instname, :cbfbytes][1]
    if filesize > 50_000_000
        println(fdmeta, "\nskipping $instname $ss due to size")
        flush(fdmeta)
        continue
    end

    println(fdmeta, "\nstarting $instname $solver $ss")
    flush(fdmeta)
    filename = joinpath(outputpath, instname * ".txt")

    try
        t = time()
        # process = run(pipeline(`$(joinpath(Sys.BINDIR, "julia")) --trace-compile=snoop scripts/run_single.jl $instname $csvfile $solver $ss`, stdout = filename, stderr = filename, append = true), wait = false)
        process = run(pipeline(`$(joinpath(Sys.BINDIR, "julia")) cblib/scripts/run_single.jl $instname $csvfile $solver $ss`, stdout = filename, stderr = filename, append = true), wait = false)
        # process = run(pipeline(`$(joinpath(Sys.BINDIR, "julia")) --trace-compile=snoop --sysimage=sc_img.so scripts/run_single.jl $instname $csvfile $solver $ss`, stdout = filename, stderr = filename, append = true), wait = false)
        sleep(3.0)
        pid = parse(Int, chomp(readline(open("mypid", "r"))))
        while process_running(process)
            if (time() - t) > (tlim + 60.0)
                # kill if time limit exceeded (some solvers don't respect time limits)
                kill(process)
                sleep(1.0)
                println(fdmeta, "killed by time limit")
                flush(fdmeta)
                open(filename, "a") do fd
                    println(fd, "#STATUS# KilledTime")
                end
                open(csvfile, "a") do c
                    print(c, "KILLED_TIME,$(repeat(",", 16))")
                end
            else
                try
                    if !process_exited(process)
                        memuse = parse(Int, split(read(pipeline(`cat /proc/$pid/status`,`grep RSS`), String))[2])
                        if memuse > mlim
                            kill(process)
                            sleep(5.0)
                            println(fdmeta, "killed by memory limit")
                            open(filename, "a") do fd
                                println(fd, "#STATUS# KilledMemory")
                            end
                            open(csvfile, "a") do c
                                print(c, "KILLED_MEMORY,$(repeat(",", 16))")
                            end
                        end
                    end
                catch e
                    println(fdmeta, "...error in memory check: $e")
                end
            end
            sleep(1.0)
        end

        println(fdmeta, "...took $(time() - t) seconds")
    catch e
        println(fdmeta, "...process error: $e")
    end
    flush(fdmeta)

end # instances, solvers
flush(fdmeta)
close(fdmeta)

println("\ndone\n")
