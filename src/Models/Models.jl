#=
Copyright 2018, Chris Coey and contributors

functions and types for model data
=#

module Models

using LinearAlgebra
using SparseArrays

import Hypatia.Cones

using TimerOutputs
import Hypatia
# import Hypatia.to

mutable struct Point
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    s::Vector{Float64}

    z_views::Vector{SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}}
    s_views::Vector{SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}}
    dual_views::Vector{SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}}
    primal_views::Vector{SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}}

    function Point(x, y, z, s, cones, cone_idxs)
        point = new()

        point.x = x
        point.y = y
        point.z = z
        point.s = s

        point.z_views = [view(point.z, idxs) for idxs in cone_idxs]
        point.s_views = [view(point.s, idxs) for idxs in cone_idxs]
        point.dual_views = [Cones.use_dual(cones[k]) ? point.s_views[k] : point.z_views[k] for k in eachindex(cones)]
        point.primal_views = [Cones.use_dual(cones[k]) ? point.z_views[k] : point.s_views[k] for k in eachindex(cones)]

        return point
    end
end

abstract type Model end

abstract type LinearModel <: Model end
include("linear.jl")

# TODO other model types eg quadratic obj, convex differentiable obj

end
