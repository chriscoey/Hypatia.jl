#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

j scripts/run_single.jl demb782 dummy_results.txt hypatia naive_dense

helper file that calls single_moi, factoring out any code that can be avoided during precompilation
=#

import Dates

# instname = "enpro56"
instname = "syn10m04m"
csvfile =  "dummy_results.txt"
solver_name = "hypatia"
system_solver_name = "qrchol_dense"
# system_solver_name = "naive_dense"
# model_type = BigFloat
model_type = Float64

println()
@show instname
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
single_moi(instname, csvfile, system_solver_name, print_timer = true, precompiling = false, out_type = model_type)

println()
