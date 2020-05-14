#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

j scripts/run_single.jl demb782 dummy_results.txt hypatia naive_dense

helper file that calls single_moi, factoring out any code that can be avoided during precompilation
=#

import Dates
include(joinpath(@__DIR__, "read_instances.jl"))

nargs = length(ARGS)
if !(nargs in [3, 4])
    error("usage: julia run_single.jl instname csvfile solver system_solver_name")
end

open("mypid", "w") do fd
    print(fd, getpid())
end

instname = ARGS[1]
csvfile = ARGS[2]
solver_name = ARGS[3]
system_solver_name = (solver_name == "hypatia" ? ARGS[4] : "")
# # bss1 solved with nbhds 1e-2
# # demb782 solved with 1e-3
# instname = "bss2"
# csvfile =  "dummy_results.txt"
# solver_name = "hypatia"
# system_solver_name = "qrchol_dense"
# system_solver_name = "naive_dense"
# model_type = BigFloat
model_type = Float64

println()
println("instance $instname")
println("ran at: ", Dates.now())
println("with solver: $solver_name, system solver: $system_solver_name")
println()
flush(stdout)

include(joinpath(@__DIR__, "single_moi.jl"))
open(csvfile, "a") do fdcsv
    print(fdcsv, "\n$instname,$model_type,$solver_name,$system_solver_name,")
    flush(fdcsv)
end
# if solver_name == "hypatia"
    # println("solving small instances to precompile...")
    # for instname in read_instances(joinpath(@__DIR__, "../sets", "compile.txt"))
    #     println("solving small instance: $instname...")
    #     flush(stdout)
    #     single_moi(instname, csvfile, system_solver_name, print_timer = false, precompiling = true)
    # end
# end
single_moi(instname, csvfile, system_solver_name, print_timer = true, precompiling = false, out_type = model_type)

println()
