#=
Copyright 2018, Chris Coey and contributors

eliminates the s row and column from the 4x4 system and performs one 3x3 linear system solve (see naive3 method)
reduces 3x3 system to a 2x2 system and solves via two sequential (dense) choleskys (see CVXOPT)
if there are no equality constraints, only one cholesky is needed

TODO are there cases where a sparse cholesky would perform better?
=#

mutable struct Chol2 <: LinSysCache
    n
    p
    q
    P
    A
    G
    cone

    function Chol2(
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
        (L.n, L.p, L.q, L.P, L.A, L.G, L.cone) = (n, p, q, P, A, G, cone)
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
    L::Chol2,
    )
    (n, p, q, P, A, G, cone) = (L.n, L.p, L.q, L.P, L.A, L.G, L.cone)

    zrhs3 = copy(zrhs)
    for k in eachindex(cone.prmtvs)
        sview = view(srhs, cone.idxs[k])
        calcHiarr_prmtv!(sview, cone.prmtvs[k])
        @. zrhs3[cone.idxs[k]] -= sview / mu
    end

    # if cone.prmtvs[k].usedual # G*x - mu*H*z = zrhs - srhs
    #     calcHarr_prmtv!(Hview, -mu*I, cone.prmtvs[k])
    #     @. @views L.rhs[idxs] -= srhs[cone.idxs[k]]
    # else # G*x - (mu*H)\z = zrhs - (mu*H)\srhs
    #     calcHiarr_prmtv!(Hview, -inv(mu)*I, cone.prmtvs[k])
    #     sview = view(srhs, cone.idxs[k])
    #     calcHiarr_prmtv!(sview, cone.prmtvs[k])
    #     @. L.rhs[idxs] -= sview / mu
    # end

    HG = Matrix{Float64}(undef, q, n)
    for k in eachindex(cone.prmtvs)
        calcHarr_prmtv!(view(HG, cone.idxs[k], :), view(G, cone.idxs[k], :), cone.prmtvs[k])
    end
    GHG = mu*G'*HG
    PGHG = Symmetric(P + GHG)
    F1 = cholesky!(PGHG, check=false) # TODO allow pivot
    singular = !issuccess(F1)

    # TODO if singular, use the fallback at bottom of s10.1
    if singular
        println("singular PGHG")
        PGHGAA = Symmetric(P + GHG + A'*A)
        F1 = cholesky!(PGHGAA, check=false) # TODO allow pivot
        if !issuccess(F1)
            error("could not fix singular PGHG")
        end
    end

    LA = copy(A')
    ldiv!(F1.L, LA)
    ALLA = Symmetric(LA'*LA)
    F2 = cholesky!(ALLA, check=false) # TODO allow pivot; TODO avoid if no equalities?
    if !issuccess(F2)
        error("singular ALLA")
    end

    Hz = similar(zrhs3)
    for k in eachindex(cone.prmtvs)
        calcHarr_prmtv!(view(Hz, cone.idxs[k]), view(zrhs3, cone.idxs[k]), cone.prmtvs[k])
    end
    xGHz = xrhs + mu*G'*Hz
    if singular
        xGHz += A'*yrhs
    end
    LxGHz = copy(xGHz)
    ldiv!(F1.L, LxGHz)

    y = LA'*LxGHz - yrhs
    ldiv!(F2, y)

    x = xGHz - A'*y
    ldiv!(F1, x)

    z = similar(zrhs3)
    Gxz = mu*(G*x - zrhs3)
    for k in eachindex(cone.prmtvs)
        calcHarr_prmtv!(view(z, cone.idxs[k]), view(Gxz, cone.idxs[k]), cone.prmtvs[k])
    end

    srhs .= zrhs # G*x + s = zrhs
    xrhs .= x
    yrhs .= y
    zrhs .= z
    srhs .-= G*x

    return
end
