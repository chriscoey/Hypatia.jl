#=
Copyright 2018, Chris Coey and contributors
=#

module Hypatia

using LinearAlgebra

const HypReal = Union{AbstractFloat, Rational}
const HypRealOrComplex{T <: HypReal} = Union{T, Complex{T}}

include("linearalgebra.jl")
const HypLinMap{T <: HypReal} = Union{UniformScaling, AbstractMatrix{T}, HypBlockMatrix{T}}

# submodules
include("ModelUtilities/ModelUtilities.jl")
include("Cones/Cones.jl")
include("Models/Models.jl")
include("Solvers/Solvers.jl")

# MathOptInterface
using SparseArrays
import MathOptInterface
const MOI = MathOptInterface

include("MathOptInterface/cones.jl")
include("MathOptInterface/wrapper.jl")

end
