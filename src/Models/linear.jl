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

TODO
- could optionally rescale rows of [A, b] and [G, h] and [A', G', c] and variables, for better numerics
=#

function initialize_cone_point(cones::Vector{<:Cones.Cone{T}}, cone_idxs::Vector{UnitRange{Int}}) where {T <: Real}
    q = isempty(cones) ? 0 : sum(Cones.dimension, cones)
    point = Point(T[], T[], Vector{T}(undef, q), Vector{T}(undef, q), cones, cone_idxs)
    for k in eachindex(cones)
        cone_k = cones[k]
        Cones.setup_data(cone_k)
        primal_k = point.primal_views[k]
        Cones.set_initial_point(primal_k, cone_k)
        Cones.load_point(cone_k, primal_k)
        @assert Cones.is_feas(cone_k)
        g = Cones.grad(cone_k)
        @. point.dual_views[k] = -g
    end
    return point
end

# NOTE (pivoted) QR factorizations are usually rank-revealing but may be unreliable, see http://www.math.sjsu.edu/~foster/rankrevealingcode.html
# TODO could replace this with rank(Ap_fact) when available for both dense and sparse
function get_rank_est(qr_fact, tol_qr::Real)
    R = qr_fact.R
    rank_est = 0
    for i in 1:size(R, 1) # TODO could replace this with rank(AG_fact) when available for both dense and sparse
        if abs(R[i, i]) > tol_qr
            rank_est += 1
        end
    end
    return rank_est
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

const sparse_QR_reals = Float64

mutable struct RawLinearModel{T <: Real} <: LinearModel{T}
    n::Int
    p::Int
    q::Int
    c::Vector{T}
    A
    b::Vector{T}
    G
    h::Vector{T}
    cones::Vector{Cones.Cone{T}}
    cone_idxs::Vector{UnitRange{Int}} # TODO allow generic Integer type for UnitRange parameter
    nu::T

    initial_point::Point{T}

    function RawLinearModel{T}(
        c::Vector{T},
        A,
        b::Vector{T},
        G,
        h::Vector{T},
        cones::Vector{Cones.Cone{T}};
        use_iterative::Bool = false,
        tol_qr::Real = 1e2 * eps(T),
        use_dense_fallback::Bool = true,
        ) where {T <: Real}
        model = new{T}()

        model.n = length(c)
        model.p = length(b)
        model.q = length(h)
        model.c = c
        model.A = A
        model.b = b
        model.G = G
        model.h = h
        model.cones = cones
        model.cone_idxs = build_cone_idxs(model.q, model.cones)
        model.nu = isempty(cones) ? zero(T) : sum(Cones.get_nu, cones)

        find_initial_point(model, use_iterative, tol_qr, use_dense_fallback)

        return model
    end
end

# get initial point for RawLinearModel
# TODO can run the x and y finding in parallel since they do not depend on eachother
function find_initial_point(model::RawLinearModel{T}, use_iterative::Bool, tol_qr::Real, use_dense_fallback::Bool) where {T <: Real}
    A = model.A
    G = model.G
    n = model.n
    p = model.p
    q = model.q

    point = initialize_cone_point(model.cones, model.cone_idxs)
    @assert q == length(point.z)

    # TODO optionally use row and col scaling on AG and apply to c, b, h
    # TODO precondition iterative methods
    # TODO pick default tol for iter methods

    # solve for x as least squares solution to Ax = b, Gx = h - s
    if !iszero(n)
        rhs = vcat(model.b, model.h - point.s)
        if use_iterative
            # use iterative solvers method TODO pick lsqr or lsmr
            AG = BlockMatrix{T}(p + q, n, [A, G], [1:p, (p + 1):(p + q)], [1:n, 1:n])
            point.x = zeros(T, n)
            IterativeSolvers.lsqr!(point.x, AG, rhs)
        else
            AG = vcat(A, G)
            if issparse(AG) && !(T <: sparse_QR_reals)
                # TODO alternative fallback is to convert sparse{T} to sparse{Float64} and do the sparse LU
                if use_dense_fallback
                    @warn("using dense factorization of [A; G] in initial point finding because sparse factorization for number type $T is not supported by SparseArrays")
                    AG = Matrix(AG)
                else
                    error("sparse factorization for number type $T is not supported by SparseArrays, so Hypatia cannot find an initial point")
                end
            end
            AG_fact = issparse(AG) ? qr(AG) : qr!(AG)

            AG_rank = get_rank_est(AG_fact, tol_qr)
            if AG_rank < n
                @warn("some dual equalities appear to be dependent; try using PreprocessedLinearModel")
            end

            point.x = AG_fact \ rhs
        end
    end

    # solve for y as least squares solution to A'y = -c - G'z
    if !iszero(p)
        rhs = -model.c - G' * point.z
        if use_iterative
            # use iterative solvers method TODO pick lsqr or lsmr
            point.y = zeros(T, p)
            IterativeSolvers.lsqr!(point.y, A', rhs)
        else
            if issparse(A) && !(T <: sparse_QR_reals)
                # TODO alternative fallback is to convert sparse{T} to sparse{Float64} and do the sparse LU
                if use_dense_fallback
                    @warn("using dense factorization of A' in initial point finding because sparse factorization for number type $T is not supported by SparseArrays")
                    Ap_fact = qr!(Matrix(A'))
                else
                    error("sparse factorization for number type $T is not supported by SparseArrays, so Hypatia cannot find an initial point")
                end
            else
                Ap_fact = issparse(A) ? qr(sparse(A')) : qr!(Matrix(A'))
            end

            Ap_rank = get_rank_est(Ap_fact, tol_qr)
            if Ap_rank < p
                @warn("some primal equalities appear to be dependent; try using PreprocessedLinearModel")
            end

            point.y = Ap_fact \ rhs
        end
    end

    model.initial_point = point

    return
end

get_original_data(model::RawLinearModel) = (model.c, model.A, model.b, model.G, model.h, model.cones, model.cone_idxs)

# TODO specialize method when A = I
mutable struct PreprocessedLinearModel{T <: Real} <: LinearModel{T}
    c_raw::Vector{T}
    A_raw
    b_raw::Vector{T}
    G_raw

    n::Int
    p::Int
    q::Int
    c::Vector{T}
    A
    b::Vector{T}
    G
    h::Vector{T}
    cones::Vector{Cones.Cone{T}}
    cone_idxs::Vector{UnitRange{Int}}
    nu::T

    x_keep_idxs::AbstractVector{Int}
    y_keep_idxs::AbstractVector{Int}
    Ap_R::UpperTriangular{T, <:AbstractMatrix{T}}
    Ap_Q::Union{UniformScaling, AbstractMatrix{T}}

    initial_point::Point

    function PreprocessedLinearModel{T}(
        c::Vector{T},
        A,
        b::Vector{T},
        G,
        h::Vector{T},
        cones::Vector{Cones.Cone{T}};
        tol_qr::Real = 1e2 * eps(T),
        use_dense_fallback::Bool = true,
        ) where {T <: Real}
        model = new{T}()

        model.c_raw = c
        model.A_raw = A
        model.b_raw = b
        model.G_raw = G
        model.q = length(h)
        model.h = h
        model.cones = cones
        model.cone_idxs = build_cone_idxs(model.q, model.cones)
        model.nu = isempty(cones) ? zero(T) : sum(Cones.get_nu, cones)

        preprocess_find_initial_point(model, tol_qr, use_dense_fallback)

        return model
    end
end

# preprocess and get initial point for PreprocessedLinearModel
function preprocess_find_initial_point(model::PreprocessedLinearModel{T}, tol_qr::Real, use_dense_fallback::Bool) where {T <: Real}
    c = model.c_raw
    A = model.A_raw
    b = model.b_raw
    G = model.G_raw
    q = model.q
    n = length(c)
    p = length(b)

    point = initialize_cone_point(model.cones, model.cone_idxs)
    @assert q == length(point.z)

    # solve for x as least squares solution to Ax = b, Gx = h - s
    if !iszero(n)
        # get pivoted QR # TODO when Julia has a unified QR interface, replace this
        AG = vcat(A, G)
        if issparse(AG) && !(T <: sparse_QR_reals)
            if use_dense_fallback
                @warn("using dense factorization of [A; G] in preprocessing and initial point finding because sparse factorization for number type $T is not supported by SparseArrays")
                AG = Matrix(AG)
            else
                error("sparse factorization for number type $T is not supported by SparseArrays, so Hypatia cannot preprocess and find an initial point")
            end
        end
        AG_fact = issparse(AG) ? qr(AG, tol = tol_qr) : qr(AG, Val(true))

        AG_rank = get_rank_est(AG_fact, tol_qr)
        if AG_rank == n
            # no dual equalities to remove
            x_keep_idxs = 1:n
            point.x = AG_fact \ vcat(b, model.h - point.s)
        else
            col_piv = (AG_fact isa QRPivoted{T, Matrix{T}}) ? AG_fact.p : AG_fact.pcol
            x_keep_idxs = col_piv[1:AG_rank]
            AG_R = UpperTriangular(AG_fact.R[1:AG_rank, 1:AG_rank])

            # TODO optimize all below
            c_sub = c[x_keep_idxs]
            yz_sub = AG_fact.Q * (Matrix{T}(I, p + q, AG_rank) * (AG_R' \ c_sub))
            if !(AG_fact isa QRPivoted{T, Matrix{T}})
                yz_sub = yz_sub[AG_fact.rpivinv]
            end
            residual = norm(AG' * yz_sub - c, Inf)
            if residual > tol_qr
                error("some dual equality constraints are inconsistent (residual $residual, tolerance $tol_qr)")
            end
            println("removed $(n - AG_rank) out of $n dual equality constraints")

            point.x = AG_R \ (Matrix{T}(I, AG_rank, p + q) * (AG_fact.Q' * vcat(b, model.h - point.s)))

            c = c_sub
            A = A[:, x_keep_idxs]
            G = G[:, x_keep_idxs]
            n = AG_rank
        end
    else
        x_keep_idxs = Int[]
    end

    # solve for y as least squares solution to A'y = -c - G'z
    if !iszero(p)
        if issparse(A) && !(T <: sparse_QR_reals)
            if use_dense_fallback
                @warn("using dense factorization of A' in preprocessing and initial point finding because sparse factorization for number type $T is not supported by SparseArrays")
                Ap_fact = qr!(Matrix(A'), Val(true))
            else
                error("sparse factorization for number type $T is not supported by SparseArrays, so Hypatia cannot preprocess and find an initial point")
            end
        else
            Ap_fact = issparse(A) ? qr(sparse(A'), tol = tol_qr) : qr(A', Val(true))
        end

        Ap_rank = get_rank_est(Ap_fact, tol_qr)

        Ap_R = UpperTriangular(Ap_fact.R[1:Ap_rank, 1:Ap_rank])
        col_piv = (Ap_fact isa QRPivoted{T, Matrix{T}}) ? Ap_fact.p : Ap_fact.pcol
        y_keep_idxs = col_piv[1:Ap_rank]
        Ap_Q = Ap_fact.Q

        # TODO optimize all below
        b_sub = b[y_keep_idxs]
        if Ap_rank < p
            # some dependent primal equalities, so check if they are consistent
            x_sub = Ap_Q * (Matrix{T}(I, n, Ap_rank) * (Ap_R' \ b_sub))
            if !(Ap_fact isa QRPivoted{T, Matrix{T}})
                x_sub = x_sub[Ap_fact.rpivinv]
            end
            residual = norm(A * x_sub - b, Inf)
            if residual > tol_qr
                error("some primal equality constraints are inconsistent (residual $residual, tolerance $tol_qr)")
            end
            println("removed $(p - Ap_rank) out of $p primal equality constraints")
        end

        point.y = Ap_R \ (Matrix{T}(I, Ap_rank, n) * (Ap_fact.Q' *  (-c - G' * point.z)))

        if !(Ap_fact isa QRPivoted{T, Matrix{T}})
            row_piv = Ap_fact.prow
            A = A[y_keep_idxs, row_piv]
            c = c[row_piv]
            G = G[:, row_piv]
            x_keep_idxs = x_keep_idxs[row_piv]
        else
            A = A[y_keep_idxs, :]
        end
        b = b_sub
        p = Ap_rank
        model.Ap_R = Ap_R
        model.Ap_Q = Ap_Q
    else
        y_keep_idxs = Int[]
        model.Ap_R = UpperTriangular(zeros(T, 0, 0))
        model.Ap_Q = I
    end

    model.c = c
    model.A = A
    model.b = b
    model.G = G
    model.n = n
    model.p = p
    model.x_keep_idxs = x_keep_idxs
    model.y_keep_idxs = y_keep_idxs

    model.initial_point = point

    return
end

get_original_data(model::PreprocessedLinearModel) = (model.c_raw, model.A_raw, model.b_raw, model.G_raw, model.h, model.cones)
