#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

import JLD
# import FileIO
import MathOptFormat
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import Hypatia
import Hypatia.Cones # TODO remove
import SparseArrays # TODO remove, for lading

instanceset = "easy"
instsetfile = "benchmark/instancesets/easy.txt"
inputpath = joinpath(@__DIR__, "instancefiles", "cbf")
outputpath = joinpath(@__DIR__, "instancefiles", "jld")
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

optimizer = MOI.Utilities.CachingOptimizer(HypatiaModelData{Float64}(), Hypatia.Optimizer())

for instname in instances
    println("opening $instname")
    fullpathin = joinpath(inputpath, instname)
    model = MathOptFormat.read_from_file(fullpathin)
    MOI.empty!(optimizer)
    MOI.copy_to(optimizer, model)
    nativedata = optimizer.optimizer
    nativedata.load_only = true
    MOI.optimize!(optimizer)
    (c, A, b, G, h, cones, cone_idxs) = (nativedata.c, nativedata.A, nativedata.b, nativedata.G, nativedata.h, nativedata.cones, nativedata.cone_idxs)
    fullpathout = joinpath(outputpath, chop(instname, tail = 4) * ".jld")
    # FileIO.save(fullpathout, "modeldata", (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs))
    JLD.save(fullpathout, "c", c, "A", A, "b", b, "G", G, "h", h, "cones", cones, "cone_idxs", cone_idxs)
end

# fullpathout = joinpath(outputpath, chop(instname, tail = 4) * ".jld2")
# md = FileIO.load(fullpathout, "modeldata")


;
