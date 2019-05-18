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

@testset "speed" begin
    test_polymin_duals(verbose = true)
    test_envelope_dual(verbose = false)
end

Hypatia.to

;
