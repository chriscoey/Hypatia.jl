#=
Copyright 2018, Chris Coey and contributors
=#

module Hypatia

const RealOrComplex{T <: Real} = Union{T, Complex{T}}

# linear algebra helpers
using LinearAlgebra
include("linearalgebra/blockmatrix.jl")
include("linearalgebra/dense.jl")
include("linearalgebra/sparse.jl")

# # optional dependencies using Requires.jl
# import Requires
# function __init__()
#     Requires.@require Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2" include("linearalgebra/Pardiso.jl")
#     Requires.@require HSL = "34c5aeac-e683-54a6-a0e9-6e0fdc586c50" include("linearalgebra/HSL.jl")
# end

# submodules
include("ModelUtilities/ModelUtilities.jl")
include("Cones/Cones.jl")
# include("Models/Models.jl")
# include("Solvers/Solvers.jl")

# # MathOptInterface helpers
# using SparseArrays
# import MathOptInterface
# const MOI = MathOptInterface
# include("MathOptInterface/cones.jl")
# include("MathOptInterface/wrapper.jl")

end
