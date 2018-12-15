#=
Copyright 2018, Chris Coey and contributors

naive method that simply performs one high-dimensional linear system solve
=#

mutable struct NaiveCache <: LinSysCache
    n
    p
    q
    P
    c
    A
    b
    G
    h
    cone
    LHS4
    LHS4copy
    rhs4
    issymm

    function NaiveCache(
        P::AbstractMatrix{Float64},
        c::Vector{Float64},
        A::AbstractMatrix{Float64},
        b::Vector{Float64},
        G::AbstractMatrix{Float64},
        h::Vector{Float64},
        cone::Cone,
        )
        L = new()
        (n, p, q) = (length(c), length(b), length(h))
        (L.n, L.p, L.q) = (n, p, q)
        L.P = P
        L.c = c
        L.A = A
        L.b = b
        L.G = G
        L.h = h
        L.cone = cone

        # tx ty tz ts
        if issparse(P) && issparse(A) && issparse(G)
            L.LHS4 = [
                P             A'            G'                spzeros(n,q)     ;
                A             spzeros(p,p)  spzeros(p,q)      spzeros(p,q)     ;
                G             spzeros(q,p)  spzeros(q,q)      sparse(1.0I,q,q) ;
                spzeros(q,n)  spzeros(q,p)  sparse(1.0I,q,q)  sparse(1.0I,q,q) ]
        else
            L.LHS4 = [
                P           A'          G'                zeros(n,q)       ;
                A           zeros(p,p)  zeros(p,q)        zeros(p,q)       ;
                G           zeros(q,p)  zeros(q,q)        Matrix(1.0I,q,q) ;
                zeros(q,n)  zeros(q,p)  Matrix(1.0I,q,q)  Matrix(1.0I,q,q) ]
        end
        L.LHS4copy = similar(L.LHS4)
        L.rhs4 = zeros(n+p+2q)
        L.issymm = !any(cone.prmtvs[k].usedual for k in eachindex(cone.prmtvs))

        return L
    end
end

# solve system for x, y, z, s
function solvelinsys4!(
    xrhs::Vector{Float64},
    yrhs::Vector{Float64},
    zrhs::Vector{Float64},
    srhs::Vector{Float64},
    mu::Float64,
    L::NaiveCache,
    )
    (cone, n, p, q) = (L.cone, L.n, L.p, L.q)

    L.rhs4[1:n] = xrhs
    L.rhs4[n+1:n+p] = yrhs
    L.rhs4[n+p+1:n+p+q] = zrhs
    L.rhs4[n+p+q+1:end] = srhs

    @. L.LHS4copy = L.LHS4

    for k in eachindex(cone.prmtvs)
        rows = (n + p + q) .+ cone.idxs[k]
        cols = cone.prmtvs[k].usedual ? (rows .- q) : rows
        Hview = view(L.LHS4copy, rows, cols)
        calcHarr_prmtv!(Hview, mu*I, cone.prmtvs[k])
    end

    if issparse(L.LHS4copy)
        F = lu(L.LHS4copy)
    elseif L.issymm
        F = bunchkaufman!(L.LHS4copy)
    else
        F = lu!(L.LHS4copy)
    end
    ldiv!(F, L.rhs4)

    @. @views begin
        xrhs = L.rhs4[1:n]
        yrhs = L.rhs4[n+1:n+p]
        zrhs = L.rhs4[n+p+1:n+p+q]
        srhs = L.rhs4[n+p+q+1:end]
    end

    return
end
