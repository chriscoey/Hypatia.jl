#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors
=#

using TimerOutputs
using Test

examples_dir = joinpath(@__DIR__, "../examples")
include(joinpath(examples_dir, "envelope/native.jl"))
# include(joinpath(examples_dir, "linearopt/native.jl"))
include(joinpath(examples_dir, "polymin/native.jl"))
# include(joinpath(examples_dir, "contraction/JuMP.jl"))
# include(joinpath(examples_dir, "densityest/JuMP.jl"))
# include(joinpath(examples_dir, "envelope/JuMP.jl"))
# include(joinpath(examples_dir, "expdesign/JuMP.jl"))
# include(joinpath(examples_dir, "lotkavolterra/JuMP.jl"))
# include(joinpath(examples_dir, "muconvexity/JuMP.jl"))
# include(joinpath(examples_dir, "polymin/JuMP.jl"))
# include(joinpath(examples_dir, "polynorm/JuMP.jl"))
# include(joinpath(examples_dir, "regionofattr/JuMP.jl"))
# include(joinpath(examples_dir, "secondorderpoly/JuMP.jl"))
# include(joinpath(examples_dir, "shapeconregr/JuMP.jl"))
# include(joinpath(examples_dir, "semidefinitepoly/JuMP.jl"))

reset_timer!(Hypatia.to)

open(joinpath("timings", "allpolymins" * ".csv"), "a") do f
    @printf(f, "\n%15s, %15s, %15s, %15s, %15s, %15s, %15s, %15s, %15s, %15s, %15s\n",
        "poly", "n", "halfdeg", "G1", "total time", "affine %", "interp time", "num iters", "aff p iter",
        "comb per iter", "dir %",
        )

end

@testset "speed" begin
    test_polymin_duals(verbose = true)
    # test_envelope_dual(verbose = true)
    # test_polymin_dual_hearts(verbose = true)
end

Hypatia.to

;
