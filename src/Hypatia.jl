#=
Copyright 2018, Chris Coey and contributors
=#

module Hypatia
    using Printf
    using LinearAlgebra
    using SparseArrays

    using LinearAlgebra: BlasInt
    include("lapack.jl")

    import FFTW
    import Combinatorics
    import GSL: sf_gamma_inc_Q
    include("interpolation.jl")

    import ForwardDiff
    import DiffResults
    include("cone.jl")
    for primitivecone in [
        "orthant",
        "epinorminf",
        "epinormeucl",
        "epipersquare",
        "hypoperlog",
        "epiperpower",
        "hypogeomean",
        "epinormspectral",
        "semidefinite",
        "wsospolyinterp",
        "hypoperlogdet",
        "epipersumexp",
        ]
        include(joinpath(@__DIR__, "primitivecones", primitivecone * ".jl"))
    end

    # import IterativeSolvers
    include("linearsystem.jl")
    for linsyssolver in [
        "chol2",
        "qrchol",
        "naive3",
        "naive4",
        ]
        include(joinpath(@__DIR__, "linsyssolvers", linsyssolver * ".jl"))
    end

    include("nativeinterface.jl")

    import MathOptInterface
    const MOI = MathOptInterface
    include("mathoptinterface.jl")
end
