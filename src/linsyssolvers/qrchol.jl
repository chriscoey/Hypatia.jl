#=
Copyright 2018, Chris Coey and contributors

eliminates the s row and column from the 4x4 system and performs one 3x3 linear system solve (see naive3 method)
requires QR-based preprocessing of A', uses resulting Q2 and RiQ1 matrices to eliminate equality constraints
uses a Cholesky to solve a reduced symmetric linear system

TODO option for solving linear system with positive definite matrix using iterative method (eg CG)
TODO refactor many common elements with chol2
=#

mutable struct QRChol <: LinSysCache
    n
    p
    q
    P
    A
    G
    Q2
    RiQ1
    cone

    function QRChol(
        P::AbstractMatrix{Float64},
        c::Vector{Float64},
        A::AbstractMatrix{Float64},
        b::Vector{Float64},
        G::AbstractMatrix{Float64},
        h::Vector{Float64},
        cone::Cone,
        Q2::AbstractMatrix{Float64},
        RiQ1::AbstractMatrix{Float64},
        )
        L = new()
        (n, p, q) = (length(c), length(b), length(h))
        (L.n, L.p, L.q, L.P, L.A, L.G, L.Q2, L.RiQ1, L.cone) = (n, p, q, P, A, G, Q2, RiQ1, cone)
        return L
    end
end

QRChol(
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    cone::Cone,
    Q2::AbstractMatrix{Float64},
    RiQ1::AbstractMatrix{Float64},
    ) = QRChol(Symmetric(spzeros(length(c), length(c))), c, A, b, G, h, cone, Q2, RiQ1)


# solve system for x, y, z, s
function solvelinsys4!(
    xrhs::Vector{Float64},
    yrhs::Vector{Float64},
    zrhs::Vector{Float64},
    srhs::Vector{Float64},
    mu::Float64,
    L::QRChol,
    )
    (n, p, q, P, A, G, Q2, RiQ1, cone) = (L.n, L.p, L.q, L.P, L.A, L.G, L.Q2, L.RiQ1, L.cone)

    # TODO refactor the conversion to 3x3 system and back (start and end)
    zrhs3 = copy(zrhs)
    for k in eachindex(cone.prmtvs)
        sview = view(srhs, cone.idxs[k])
        zview = view(zrhs3, cone.idxs[k])
        if cone.prmtvs[k].usedual # G*x - mu*H*z = zrhs - srhs
            zview .-= sview
        else # G*x - (mu*H)\z = zrhs - (mu*H)\srhs
            calcHiarr_prmtv!(sview, cone.prmtvs[k])
            @. zview -= sview * mu
        end
    end

    HG = Matrix{Float64}(undef, q, n)
    for k in eachindex(cone.prmtvs)
        Gview = view(G, cone.idxs[k], :)
        HGview = view(HG, cone.idxs[k], :)
        if cone.prmtvs[k].usedual
            calcHiarr_prmtv!(HGview, Gview, cone.prmtvs[k])
            HGview .*= mu
        else
            calcHarr_prmtv!(HGview, Gview, cone.prmtvs[k])
            HGview ./= mu
        end
    end
    GHG = Symmetric(G'*HG)
    PGHG = Symmetric(Symmetric(P) + GHG)
    Q2PGHGQ2 = Symmetric(Q2'*PGHG*Q2)
    # F = cholesky!(Q2PGHGQ2, Val(true), check=false)
    # singular = !isposdef(F)
    F = bunchkaufman!(Q2PGHGQ2, true, check=false)
    singular = !issuccess(F)

    if singular
        println("singular Q2PGHGQ2")
        Q2PGHGQ2 = Symmetric(Q2'*Symmetric(PGHG + Symmetric(A'*A))*Q2)
        # @show eigvals(Q2PGHGQ2)
        # F = cholesky!(Q2PGHGQ2, Val(true), check=false)
        # if !isposdef(F)
        #     error("could not fix singular Q2PGHGQ2")
        # end
        F = bunchkaufman!(Q2PGHGQ2, true, check=false)
        if !issuccess(F)
            error("could not fix singular Q2PGHGQ2")
        end
    end

    Hz = similar(zrhs3)
    for k in eachindex(cone.prmtvs)
        zview = view(zrhs3, cone.idxs[k], :)
        Hzview = view(Hz, cone.idxs[k], :)
        if cone.prmtvs[k].usedual
            calcHiarr_prmtv!(Hzview, zview, cone.prmtvs[k])
            Hzview .*= mu
        else
            calcHarr_prmtv!(Hzview, zview, cone.prmtvs[k])
            Hzview ./= mu
        end
    end
    xGHz = xrhs + G'*Hz
    if singular
        xGHz += A'*yrhs # TODO should this be minus
    end

    x = RiQ1'*yrhs
    Q2div = Q2'*(xGHz - GHG*x)
    ldiv!(F, Q2div)
    x += Q2*Q2div

    y = RiQ1*(xGHz - GHG*x)

    z = similar(zrhs3)
    Gxz = G*x - zrhs3
    for k in eachindex(cone.prmtvs)
        Gxzview = view(Gxz, cone.idxs[k], :)
        zview = view(z, cone.idxs[k], :)
        if cone.prmtvs[k].usedual
            calcHiarr_prmtv!(zview, Gxzview, cone.prmtvs[k])
            zview .*= mu
        else
            calcHarr_prmtv!(zview, Gxzview, cone.prmtvs[k])
            zview ./= mu
        end
    end

    srhs .= zrhs # G*x + s = zrhs
    xrhs .= x
    yrhs .= y
    zrhs .= z
    srhs .-= G*x

    return
end
