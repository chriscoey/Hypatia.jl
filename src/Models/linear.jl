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

The primal-dual optimality conditions are:
```
         b - Ax == 0
         h - Gx == s
  c + A'y + G'z == 0
            s'z == 0
              s in K
              z in K*
```
=#

# TODO check model data consistency

mutable struct RawLinearModel <: LinearModel
    n::Int
    p::Int
    q::Int
    c::Vector{Float64}
    A::AbstractMatrix{Float64}
    b::Vector{Float64}
    G::AbstractMatrix{Float64}
    h::Vector{Float64}
    cones::Vector{Cones.Cone}
    cone_idxs::Vector{UnitRange{Int}}
    nu::Float64

    initial_point::Point

    function RawLinearModel(c::Vector{Float64}, A::AbstractMatrix{Float64}, b::Vector{Float64}, G::AbstractMatrix{Float64}, h::Vector{Float64}, cones::Vector{<:Cones.Cone}, cone_idxs::Vector{UnitRange{Int}})
        model = new()

        model.n = length(c)
        model.p = length(b)
        model.q = length(h)
        model.c = c
        model.A = A
        model.b = b
        model.G = G
        model.h = h
        model.cones = cones
        model.cone_idxs = cone_idxs
        model.nu = isempty(cones) ? 0.0 : sum(Cones.get_nu, cones)

        # get initial point
        point = Point(Float64[], Float64[], similar(h), similar(h), cones, cone_idxs)
        set_initial_cone_point(point, model.cones)

        # solve for y as least squares solution to A'y = -c - G'z
        if !iszero(model.p)
            point.y = (issparse(A) ? sparse(A') : A') \ (-c - G' * point.z)
        end

        # solve for x as least squares solution to Ax = b, Gx = h - s
        if !iszero(model.n)
            point.x = vcat(A, G) \ vcat(b, h - point.s)
        end

        model.initial_point = point

        return model
    end
end

mutable struct PreprocessedLinearModel <: LinearModel
    c_raw::Vector{Float64}
    A_raw::AbstractMatrix{Float64}
    b_raw::Vector{Float64}
    G_raw::AbstractMatrix{Float64}

    n::Int
    p::Int
    q::Int
    c::Vector{Float64}
    A::AbstractMatrix{Float64}
    b::Vector{Float64}
    G::AbstractMatrix{Float64}
    h::Vector{Float64}
    cones::Vector{Cones.Cone}
    cone_idxs::Vector{UnitRange{Int}}
    nu::Float64

    x_keep_idxs::AbstractVector{Int}
    y_keep_idxs::AbstractVector{Int}
    Ap_R::AbstractMatrix{Float64}
    Ap_Q1::AbstractMatrix{Float64}
    Ap_Q2

    initial_point::Point

    # TODO could optionally rescale rows of [A, b] and [G, h] and [A', G', c] and variables
    # NOTE (pivoted) QR factorizations are usually rank-revealing but may be unreliable, see http://www.math.sjsu.edu/~foster/rankrevealingcode.html
    function PreprocessedLinearModel(c::Vector{Float64}, A::AbstractMatrix{Float64}, b::Vector{Float64}, G::AbstractMatrix{Float64}, h::Vector{Float64}, cones::Vector{<:Cones.Cone}, cone_idxs::Vector{UnitRange{Int}}; tol_QR::Float64 = 1e-13)
        model = new()
        model.c_raw = c
        model.A_raw = A
        model.b_raw = b
        model.G_raw = G
        n = length(c)
        p = length(b)
        q = length(h)
        model.q = q
        model.h = h
        model.cones = cones
        model.cone_idxs = cone_idxs
        model.nu = isempty(cones) ? 0.0 : sum(Cones.get_nu, cones)

        # get initial point and preprocess
        point = Point(Float64[], Float64[], similar(h), similar(h), cones, cone_idxs)
        set_initial_cone_point(point, model.cones)

        # solve for x as least squares solution to Ax = b, Gx = h - s
        if !iszero(n)
            # get pivoted QR # TODO when Julia has a unified QR interface, replace this
            AG = vcat(A, G)
            if issparse(AG)
                AG_fact = qr(AG, tol = tol_QR)
            else
                AG_fact = qr(AG, Val(true))
            end
            AG_R = AG_fact.R

            # TODO could replace this with rank(Ap_fact) when available for both dense and sparse
            AG_rank = 0
            for i in 1:size(AG_R, 1) # TODO could replace this with rank(AG_fact) when available for both dense and sparse
                if abs(AG_R[i, i]) > tol_QR
                    AG_rank += 1
                end
            end

            if AG_rank == n
                # no dual equalities to remove
                x_keep_idxs = 1:n
                point.x = AG_fact \ vcat(b, h - point.s)
            else
                # TODO optimize all below
                if issparse(AG)
                    x_keep_idxs = AG_fact.pcol[1:AG_rank]
                    AG_Q1 = Matrix{Float64}(undef, p + q, AG_rank)
                    AG_Q1[AG_fact.prow, :] = AG_fact.Q * Matrix{Float64}(I, p + q, AG_rank)
                else
                    x_keep_idxs = AG_fact.p[1:AG_rank]
                    AG_Q1 = AG_fact.Q * Matrix{Float64}(I, p + q, AG_rank)
                end
                AG_R = UpperTriangular(AG_R[1:AG_rank, 1:AG_rank])

                c_sub = c[x_keep_idxs]
                yz_sub = AG_Q1 * (AG_R' \ c_sub)
                if norm(AG' * yz_sub - c, Inf) > tol_QR
                    error("some dual equality constraints are inconsistent")
                end
                println("removed $(n - AG_rank) out of $n dual equality constraints")
                c = c_sub
                A = A[:, x_keep_idxs]
                G = G[:, x_keep_idxs]
                n = AG_rank
                point.x = AG_R \ (AG_Q1' * vcat(b, h - point.s))
            end
        else
            x_keep_idxs = Int[]
        end

        # solve for y as least squares solution to A'y = -c - G'z
        if !iszero(p)
            # get pivoted QR # TODO when Julia has a unified QR interface, replace this
            if issparse(A)
                Ap_fact = qr(sparse(A'), tol = tol_QR)
            else
                Ap_fact = qr(A', Val(true))
            end
            Ap_R = Ap_fact.R

            # TODO could replace this with rank(Ap_fact) when available for both dense and sparse
            Ap_rank = 0
            for i in 1:size(Ap_R, 1)
                if abs(Ap_R[i, i]) > tol_QR
                    Ap_rank += 1
                end
            end

            # TODO optimize all below
            if issparse(A)
                y_keep_idxs = Ap_fact.pcol[1:Ap_rank]
                A_Q = Matrix{Float64}(undef, n, n)
                A_Q[Ap_fact.prow, :] = Ap_fact.Q * Matrix{Float64}(I, n, n)
            else
                y_keep_idxs = Ap_fact.p[1:Ap_rank]
                A_Q = Ap_fact.Q * Matrix{Float64}(I, n, n)
            end
            Ap_Q1 = A_Q[:, 1:Ap_rank]
            Ap_Q2 = A_Q[:, (Ap_rank + 1):n]
            Ap_R = UpperTriangular(Ap_R[1:Ap_rank, 1:Ap_rank])

            b_sub = b[y_keep_idxs]
            if Ap_rank < p
                # some dependent primal equalities, so check if they are consistent
                x_sub = Ap_Q1 * (Ap_R' \ b_sub)
                if norm(A * x_sub - b, Inf) > tol_QR
                    error("some primal equality constraints are inconsistent")
                end
                println("removed $(p - Ap_rank) out of $p primal equality constraints")
            end
            A = A[y_keep_idxs, :]
            b = b_sub
            p = Ap_rank
            point.y = Ap_R \ (Ap_Q1' * (-c - G' * point.z)) # TODO remove allocs
            model.Ap_R = Ap_R
            model.Ap_Q1 = Ap_Q1
            model.Ap_Q2 = Ap_Q2
        else
            y_keep_idxs = Int[]
            model.Ap_R = UpperTriangular(zeros(0, 0))
            model.Ap_Q1 = zeros(n, 0)
            model.Ap_Q2 = I
        end

        model.n = n
        model.p = p
        model.c = c
        model.A = A
        model.b = b
        model.G = G
        model.x_keep_idxs = x_keep_idxs
        model.y_keep_idxs = y_keep_idxs

        model.initial_point = point

        return model
    end
end

function set_initial_cone_point(point, cones)
    for k in eachindex(cones)
        cone_k = cones[k]
        Cones.setup_data(cone_k)
        primal_k = point.primal_views[k]
        Cones.set_initial_point(primal_k, cone_k)
        Cones.load_point(cone_k, primal_k)
        @assert Cones.check_in_cone(cone_k)
        g = Cones.grad(cone_k)
        @. point.dual_views[k] = -g
    end
    return point
end

get_original_data(model::RawLinearModel) = (model.c, model.A, model.b, model.G, model.h, model.cones, model.cone_idxs)
get_original_data(model::PreprocessedLinearModel) = (model.c_raw, model.A_raw, model.b_raw, model.G_raw, model.h, model.cones, model.cone_idxs)
