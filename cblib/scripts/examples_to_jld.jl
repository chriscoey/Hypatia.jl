#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

julia scripts/examples_to_jld.jl JuMP_easy JuMP instances
julia scripts/examples_to_jld.jl native_easy native instances
=#

# import Pkg
# Pkg.activate(".")

import JLD
import Random
import JuMP
import Hypatia
include(joinpath(@__DIR__, "read_instances.jl"))

println()
if length(ARGS) != 3
    error("usage: julia examples_to_jld.jl set format outputpath")
end

set = ARGS[1]
setfile = joinpath(@__DIR__, "../sets", set * ".txt")

format = ARGS[2]
if format == "native"
    isnative = true
elseif format == "JuMP"
    isnative = false
else
    error("unrecognized format: $format")
end

outputpath = ARGS[3]
if !isdir(outputpath)
    mkdir(outputpath)
end

Random.seed!(1234) # would make more sense for random seed to be in all the model functions, not outside

# includes all examples
examples_dir = abspath(joinpath(dirname(Base.find_package("Hypatia")), "../examples"))
for e in readdir(examples_dir)
    e_dir = joinpath(examples_dir, e)
    if !isdir(e_dir)
        continue
    end
    ef = joinpath(e_dir, format * ".jl")
    if isfile(ef)
        include(ef)
    end
end

function example_to_JLD(modelname::AbstractString, isnative::Bool)
    println("\nconverting $(modelname)\n")
    d = eval(Meta.parse(modelname))()
    if isnative
        (c, A, b, G, h, cones, cone_idxs) = (d.c, d.A, d.b, d.G, d.h, d.cones)
    else
        model = d.model
        JuMP.optimize!(model, JuMP.with_optimizer(Hypatia.Optimizer, load_only = true))
        opt = model.moi_backend.optimizer.model.optimizer
        (c, A, b, G, h, cones, cone_idxs) = (opt.c, opt.A, opt.b, opt.G, opt.h, opt.cones)
    end
    fullpathout = joinpath(outputpath, modelname * ".jld")
    JLD.save(fullpathout, "c", c, "A", A, "b", b, "G", G, "h", h, "cones", cones)
    return
end

instances = read_instances(setfile)
example_to_JLD.(instances, isnative)
