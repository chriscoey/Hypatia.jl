#=
Copyright 2018, Chris Coey and contributors

functions and types for linear objective conic problems of the form:

primal (over x,s):
```
  min  c'x :          duals
    b - Ax == 0       (y)
    h - Gx == s in K  (z)
```
dual (over z,y):
```
  max  -b'y - h'z :      duals
    c + A'y + G'z == 0   (x)
                z in K*  (s)
```
where K is a convex cone defined as a Cartesian product of recognized primitive cones, and K* is its dual cone.
An objective offset can be provided as the keyword arg `obj_offset` (default 0).

The primal-dual optimality conditions are:
```
         b - Ax == 0
         h - Gx == s
  c + A'y + G'z == 0
            s'z == 0
              s in K
              z in K*
```

TODO
- could optionally rescale rows of [A, b] and [G, h] and [A', G', c] and variables, for better numerics
=#

module Models

using LinearAlgebra
import Hypatia.Cones

mutable struct Point{T <: Real}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    s::Vector{T}

    z_views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
    s_views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
    dual_views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
    primal_views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}

    function Point(
        x::Vector{T},
        y::Vector{T},
        z::Vector{T},
        s::Vector{T},
        cones::Vector{<:Cones.Cone{T}},
        cone_idxs::Vector{UnitRange{Int}},
        ) where {T <: Real}
        point = new{T}()

        point.x = x
        point.y = y
        point.z = z
        point.s = s

        point.z_views = [view(point.z, idxs) for idxs in cone_idxs]
        point.s_views = [view(point.s, idxs) for idxs in cone_idxs]
        point.dual_views = [Cones.use_dual_barrier(cones[k]) ? point.s_views[k] : point.z_views[k] for k in eachindex(cones)]
        point.primal_views = [Cones.use_dual_barrier(cones[k]) ? point.z_views[k] : point.s_views[k] for k in eachindex(cones)]

        return point
    end
end

mutable struct Model{T <: Real}
    n::Int
    p::Int
    q::Int
    obj_offset::T
    c::Vector{T}
    A
    b::Vector{T}
    G
    h::Vector{T}
    cones::Vector{Cones.Cone{T}}
    cone_idxs::Vector{UnitRange{Int}}
    nu::T

    function Model{T}(
        c::Vector{T},
        A,
        b::Vector{T},
        G,
        h::Vector{T},
        cones::Vector{Cones.Cone{T}};
        obj_offset::T = zero(T),
        rescale::Bool = false, # rescale problem data # TODO un-scale in residuals (optional) and before returning solution
        ) where {T <: Real}
        model = new{T}()

        model.n = length(c)
        model.p = length(b)
        model.q = length(h)
        model.obj_offset = obj_offset
        if rescale
            # TODO might need to move to solver because model gets reformed a lot in some scripts
            # @show norm(A)
            # @show norm(G)
            rteps = sqrt(eps(T))
            c_scale = T[sqrt(max(rteps, abs(c[j]), maximum(abs, A[:, j]), maximum(abs, G[:, j]))) for j in 1:model.n]
            b_scale = T[sqrt(max(rteps, abs(b[i]), maximum(abs, A[i, :]))) for i in 1:model.p]
            h_scale = T[sqrt(max(rteps, abs(h[i]), maximum(abs, G[i, :]))) for i in 1:model.q]
            # c_mat = Diagonal(c_scale)
            # b_mat = Diagonal(b_scale)
            # h_mat = Diagonal(h_scale)
            c = c ./ c_scale
            b = b ./ b_scale
            h = h ./ h_scale
            A = A ./ c_scale' ./ b_scale
            G = G ./ c_scale' ./ h_scale
            # @show norm(A)
            # @show norm(G)
        end
        model.c = c
        model.A = A
        model.b = b
        model.G = G
        model.h = h
        model.cones = cones
        model.cone_idxs = build_cone_idxs(model.q, model.cones)
        model.nu = isempty(cones) ? zero(T) : sum(Cones.get_nu, cones)

        return model
    end
end

function build_cone_idxs(q::Int, cones::Vector{Cones.Cone{T}}) where {T <: Real}
    cone_idxs = Vector{UnitRange{Int}}(undef, length(cones))
    prev_idx = 0
    for (k, cone) in enumerate(cones)
        dim = Cones.dimension(cone)
        cone_idxs[k] = (prev_idx + 1):(prev_idx + dim)
        prev_idx += dim
    end
    @assert q == prev_idx
    return cone_idxs
end

get_cone_idxs(model::Model) = model.cone_idxs

# make the model's A and G matrices dense
function densify!(model::Model{T}) where {T <: Real}
    model.A = convert(Matrix{T}, model.A)
    model.G = convert(Matrix{T}, model.G)
    return model
end

end
