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

for nbhd in ["infty", "hess"]
    open(joinpath("timings", "allpolymins_" * nbhd * ".csv"), "a") do f
        @printf(f, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
        "poly", "obj", "n", "halfdeg", "G dim", "nu", "interp t", "build t", "solve t", "affine %t", "comb %t", "dir %t", "# iters", "# corr steps", "aff / iter",
        "comb / iter",
        )
    end
end

@testset "speed" begin
    test_polymin_duals(verbose = true)
    # test_envelope_dual(verbose = true)
    # test_polymin_dual_hearts(verbose = true)
end

Hypatia.to

;
