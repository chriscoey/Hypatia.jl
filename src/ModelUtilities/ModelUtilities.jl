#=
Copyright 2018, Chris Coey and contributors

utilities for constructing Hypatia native models and PolyJuMP.jl models
=#

module ModelUtilities

include("domains.jl")

using LinearAlgebra
import FFTW
import Combinatorics
import GSL: sf_gamma_inc_Q
import DynamicPolynomials
const DP = DynamicPolynomials
include("interpolate.jl")

import SemialgebraicSets
const SAS = SemialgebraicSets
include("semialgebraicsets.jl")

end
