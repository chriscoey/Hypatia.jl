#=
Copyright 2018, Chris Coey and contributors

naive method that eliminates the s row and column from the 4x4 system and performs one 3x3 linear system solve

G*x + s = zrhs
z + mu*H*s = srhs --> s = (mu*H)\(srhs - z)  (primal barrier)
mu*H*z + s = srhs --> s = srhs - mu*H*z     (dual barrier)
-->
G*x - (mu*H)\z = zrhs - (mu*H)\srhs  (primal barrier)
G*x - mu*H*z = zrhs - srhs           (dual barrier)
=#

mutable struct Naive3 <: LinSysCache
    n
    p
    q
    G
    cone
    LHS
    LHScopy
    rhs

    function Naive3(
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
        (L.n, L.p, L.q, L.G, L.cone) = (n, p, q, G, cone)
        if issparse(P) && issparse(A) && issparse(G)
            L.LHS = [
                P  A'            G'                ;
                A  spzeros(p,p)  spzeros(p,q)      ;
                G  spzeros(q,p)  sparse(1.0I,q,q)  ]
        else
            L.LHS = [
                Matrix(P)  Matrix(A')  Matrix(G')        ;
                Matrix(A)  zeros(p,p)  zeros(p,q)        ;
                Matrix(G)  zeros(q,p)  Matrix(1.0I,q,q)  ]
        end
        L.LHScopy = similar(L.LHS)
        L.rhs = zeros(n+p+q)
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
    L::Naive3,
    )
    (n, p, q, G, cone) = (L.n, L.p, L.q, L.G, L.cone)

    L.rhs[1:n] = xrhs
    L.rhs[n+1:n+p] = yrhs
    L.rhs[n+p+1:end] = zrhs

    @. L.LHScopy = L.LHS

    for k in eachindex(cone.prmtvs)
        idxs = (n + p) .+ cone.idxs[k]
        Hview = view(L.LHScopy, idxs, idxs)
        if cone.prmtvs[k].usedual # G*x - mu*H*z = zrhs - srhs
            calcHarr_prmtv!(Hview, -mu*I, cone.prmtvs[k])
            @. @views L.rhs[idxs] -= srhs[cone.idxs[k]]
        else # G*x - (mu*H)\z = zrhs - (mu*H)\srhs
            calcHiarr_prmtv!(Hview, -inv(mu)*I, cone.prmtvs[k])
            sview = view(srhs, cone.idxs[k])
            calcHiarr_prmtv!(sview, cone.prmtvs[k])
            @. L.rhs[idxs] -= sview / mu
        end
    end

    if issparse(L.LHScopy)
        F = ldlt(L.LHScopy)
    else
        F = bunchkaufman!(L.LHScopy)
    end
    ldiv!(F, L.rhs)

    srhs .= zrhs # G*x + s = zrhs
    @. @views begin
        xrhs = L.rhs[1:n]
        yrhs = L.rhs[n+1:n+p]
        zrhs = L.rhs[n+p+1:end]
    end
    srhs .-= G*xrhs # TODO allocates

    return
end
