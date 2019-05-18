#=
Copyright 2018, Chris Coey and contributors
=#

module Hypatia

using TimerOutputs
const to = TimerOutput()

# submodules
include("ModelUtilities/ModelUtilities.jl")
include("Cones/Cones.jl")
include("Models/Models.jl")
include("Solvers/Solvers.jl")

# MathOptInterface
using Test
using LinearAlgebra
using SparseArrays
import MathOptInterface
const MOI = MathOptInterface

include("MathOptInterface/cones.jl")
include("MathOptInterface/wrapper.jl")

end
