using Revise
include(joinpath(@__DIR__, "read_instances.jl"))
include(joinpath(@__DIR__, "single_hypatia.jl"))
setfile = joinpath(@__DIR__, "../sets", "sample.txt")
instances = ["10_0_1_w"] # read_instances(setfile) as_conic_frozenpizza_2_cap10 10_0_1_w

for instname in instances, system_solver in ["naiveelim"]
    csvfile = joinpath("sample", instname * ".csv")

    # run each instance twice
    for _ in 1:1
        single_moi(instname, csvfile, "hypatia", system_solver, print_timer = true)
    end
end
