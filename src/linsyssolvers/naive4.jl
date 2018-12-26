#=
Copyright 2018, Chris Coey and contributors

naive method that simply performs one 4x4 linear system solve
=#

mutable struct Naive4 <: LinSysCache
    n
    p
    q
    cone
    LHS
    LHScopy
    rhs
    issymm

    function Naive4(
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
        (L.n, L.p, L.q, L.cone) = (n, p, q, cone)
        if issparse(P) && issparse(A) && issparse(G)
            L.LHS = [
                P             A'            G'                spzeros(n,q)     ;
                A             spzeros(p,p)  spzeros(p,q)      spzeros(p,q)     ;
                G             spzeros(q,p)  spzeros(q,q)      sparse(1.0I,q,q) ;
                spzeros(q,n)  spzeros(q,p)  sparse(1.0I,q,q)  sparse(1.0I,q,q) ]
        else
            L.LHS = [
                P           A'          G'                zeros(n,q)       ;
                A           zeros(p,p)  zeros(p,q)        zeros(p,q)       ;
                G           zeros(q,p)  zeros(q,q)        Matrix(1.0I,q,q) ;
                zeros(q,n)  zeros(q,p)  Matrix(1.0I,q,q)  Matrix(1.0I,q,q) ]
        end
        L.LHScopy = similar(L.LHS)
        L.rhs = zeros(n+p+2q)
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
    L::Naive4,
    )
    (n, p, q, cone) = (L.n, L.p, L.q, L.cone)

    L.rhs[1:n] = xrhs
    L.rhs[n+1:n+p] = yrhs
    L.rhs[n+p+1:n+p+q] = zrhs
    L.rhs[n+p+q+1:end] = srhs

    @. L.LHScopy = L.LHS

    for k in eachindex(cone.prmtvs)
        rows = (n + p + q) .+ cone.idxs[k]
        cols = cone.prmtvs[k].usedual ? (rows .- q) : rows
        # Hview = view(L.LHScopy, rows, cols)
        # calcHarr_prmtv!(Hview, mu*I, cone.prmtvs[k])
        L.LHScopy[rows, cols] = Symmetric(cone.prmtvs[k].H) / mu
    end

    if issparse(L.LHScopy)
        F = lu(L.LHScopy)
    elseif L.issymm
        F = bunchkaufman!(L.LHScopy)
    else
        F = lu!(L.LHScopy)
    end
    ldiv!(F, L.rhs)

    @. @views begin
        xrhs = L.rhs[1:n]
        yrhs = L.rhs[n+1:n+p]
        zrhs = L.rhs[n+p+1:n+p+q]
        srhs = L.rhs[n+p+q+1:end]
    end

    return
end
