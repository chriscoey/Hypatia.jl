import CSV
include("read_instances.jl")
cblib_dir = joinpath(ENV["HOME"], "cblib/cblib.zib.de/download/all")

function make_all()
    f = open(joinpath(@__DIR__(), "../sets", "all.txt"), "w")
    for setfile in readdir(cblib_dir)
        if endswith(setfile, "cbf.gz")
            instname = chop(setfile, tail = 7)
            println(f, instname)
        end
    end
    close(f)
    return
end

# empty set
function make_orthant()
    stats = CSV.read(joinpath(@__DIR__(), "../stats/STATS_all.csv"))
    f = open(joinpath(@__DIR__(), "../sets", "orthant.txt"), "w")
    for i in 1:size(stats, 1)
        if stats[i, :semidefinitereal][1] == 0 && stats[i, :power][1] == 0 && stats[i, :epinormeucl][1] == 0 &&
            stats[i, :epipersquare][1] == 0 && stats[i, :hypoperlog][1] == 0
            instname = stats[i, :instname]
            println(f, instname)
        end
    end
    close(f)
    return
end

function make_epinormecl()
    stats = CSV.read(joinpath(@__DIR__(), "../stats/STATS_all.csv"))
    f = open(joinpath(@__DIR__(), "../sets", "epinormeucl.txt"), "w")
    for i in 1:size(stats, 1)
        if stats[i, :semidefinitereal][1] == 0 && stats[i, :power][1] == 0 &&
            stats[i, :epipersquare][1] == 0 && stats[i, :hypoperlog][1] == 0
            instname = stats[i, :instname]
            println(f, instname)
        end
    end
    close(f)
    return
end
