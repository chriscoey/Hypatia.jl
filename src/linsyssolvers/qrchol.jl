#=
Copyright 2018, Chris Coey and contributors

solve two symmetric linear systems and combine solutions (inspired by CVXOPT)
QR plus either Cholesky factorization or iterative conjugate gradients method
(1) eliminate equality constraints via QR of A'
(2) solve reduced symmetric system by Cholesky or iterative method
=#

mutable struct QRChol <: LinSysCache
    # TODO can remove some of the prealloced arrays after github.com/JuliaLang/julia/issues/23919 is resolved
    useiterative
    userefine

    cone
    c
    b
    G
    h
    Q2
    RiQ1

    bxGHbz
    Q1x
    GQ1x
    HGQ1x
    GHGQ1x
    Q2div
    GQ2
    HGQ2
    Q2GHGQ2
    Q2x
    Gxi
    HGxi
    GHGxi

    zi
    yi
    xi

    Q2divcopy
    lsferr
    lsberr
    lswork
    lsiwork
    lsAF
    lsS
    ipiv

    # cgstate
    # lprecond
    # Q2sol

    function QRChol(
        c::Vector{Float64},
        A::AbstractMatrix{Float64},
        b::Vector{Float64},
        G::AbstractMatrix{Float64},
        h::Vector{Float64},
        cone::Cone,
        Q2::AbstractMatrix{Float64},
        RiQ1::AbstractMatrix{Float64};
        useiterative::Bool = false,
        userefine::Bool = false,
        )
        @assert !useiterative # TODO disabled for now

        L = new()
        (n, p, q) = (length(c), length(b), length(h))
        nmp = n - p

        L.useiterative = useiterative
        L.userefine = userefine

        L.cone = cone
        L.c = c
        L.b = b
        L.G = G
        L.h = h
        L.Q2 = Q2
        L.RiQ1 = RiQ1

        L.bxGHbz = Matrix{Float64}(undef, n, 2)
        L.Q1x = similar(L.bxGHbz)
        L.GQ1x = Matrix{Float64}(undef, q, 2)
        L.HGQ1x = similar(L.GQ1x)
        L.GHGQ1x = Matrix{Float64}(undef, n, 2)
        L.Q2div = Matrix{Float64}(undef, nmp, 2)
        L.GQ2 = G*Q2
        L.HGQ2 = similar(L.GQ2)
        L.Q2GHGQ2 = Matrix{Float64}(undef, nmp, nmp)
        L.Q2x = similar(L.Q1x)
        L.Gxi = similar(L.GQ1x)
        L.HGxi = similar(L.Gxi)
        L.GHGxi = similar(L.GHGQ1x)

        L.zi = Matrix{Float64}(undef, q, 2)
        L.yi = Matrix{Float64}(undef, p, 2)
        L.xi = Matrix{Float64}(undef, n, 2)

        # for linear system solve with refining
        L.Q2divcopy = similar(L.Q2div)
        L.lsferr = Vector{Float64}(undef, 2)
        L.lsberr = Vector{Float64}(undef, 2)
        L.lsAF = Matrix{Float64}(undef, nmp, nmp)
        # sysvx
        L.lswork = Vector{Float64}(undef, 5*nmp)
        L.lsiwork = Vector{BlasInt}(undef, nmp)
        L.ipiv = Vector{BlasInt}(undef, nmp)
        # # posvx
        # L.lswork = Vector{Float64}(undef, 3*nmp)
        # L.lsiwork = Vector{BlasInt}(undef, nmp)
        # L.lsS = Vector{Float64}(undef, nmp)

        # # for iterative only
        # if useiterative
        #     cgu = zeros(nmp)
        #     L.cgstate = IterativeSolvers.CGStateVariables{Float64, Vector{Float64}}(cgu, similar(cgu), similar(cgu))
        #     L.lprecond = IterativeSolvers.Identity()
        #     L.Q2sol = zeros(nmp)
        # end

        return L
    end
end


QRChol(
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    cone::Cone;
    useiterative::Bool = false,
    userefine::Bool = false,
    ) = error("to use a QRChol for linear system solves, the data must be preprocessed and Q2 and RiQ1 must be passed into the QRChol constructor")


# solve symmetric 3x3 system for x, y, z
# currently assumes H is identity
function solvelinsys3!(
    rhs_tx::Vector{Float64},
    rhs_ty::Vector{Float64},
    rhs_tz::Vector{Float64},
    mu::Float64,
    L::QRChol,
    )

    # @. rhs_ty = -rhs_ty
    # @. rhs_tz = -rhs_tz

    # # bxGHbz = bx + G'*Hbz
    # mul!(L.bxGHbz, L.G', rhs_tz)
    # @. L.bxGHbz += rhs_tx

    # # Q1x = Q1*Ri'*by
    # mul!(L.Q1x, L.RiQ1', yi)

    # # Q2x = Q2*(K22_F\(Q2'*(bxGHbz - GHG*Q1x)))
    # mul!(L.GQ1x, L.G, L.Q1x)
    # for k in eachindex(L.cone.prmtvs)
    #     a1k = view(L.GQ1x, L.cone.idxs[k], :)
    #     a2k = view(L.HGQ1x, L.cone.idxs[k], :)
    #     if L.cone.prmtvs[k].usedual
    #         calcHiarr_prmtv!(a2k, a1k, L.cone.prmtvs[k])
    #         a2k ./= mu
    #     else
    #         calcHarr_prmtv!(a2k, a1k, L.cone.prmtvs[k])
    #         a2k .*= mu
    #     end
    # end
    # mul!(L.GHGQ1x, L.G', L.HGQ1x)
    # @. L.GHGQ1x = L.bxGHbz - L.GHGQ1x
    # mul!(L.Q2div, L.Q2', L.GHGQ1x)

    # # xi = Q1x + Q2x
    # @. rhs_tx = Q1x + L.Q2 * Q2div

    # # yi = Ri*Q1'*(bxGHbz - GHG*xi)
    # mul!(L.Gxi, L.G, rhs_tx)
    # for k in eachindex(L.cone.prmtvs)
    #     a1k = view(L.Gxi, L.cone.idxs[k], :)
    #     a2k = view(L.HGxi, L.cone.idxs[k], :)
    #     if L.cone.prmtvs[k].usedual
    #         calcHiarr_prmtv!(a2k, a1k, L.cone.prmtvs[k])
    #         a2k ./= mu
    #     else
    #         calcHarr_prmtv!(a2k, a1k, L.cone.prmtvs[k])
    #         a2k .*= mu
    #     end
    # end
    # mul!(L.GHGxi, L.G', L.HGxi)
    # @. L.bxGHbz -= L.GHGxi
    # mul!(yi, L.RiQ1, L.bxGHbz)

    # # zi = HG*xi - Hbz
    # @. rhs_tz = L.HGxi - rhs_tz


    BLAS.gemv!('T', -1.0, L.G, rhs_tz, 1.0, rhs_tx)

    Qx = BLAS.gemv('T', -1.0, L.RiQ1, rhs_ty)

    Q2div = L.Q2' * (rhs_tx - L.G' * L.G * Qx)

    if size(L.Q2div, 1) > 0
        BLAS.syrk!('U', 'T', 1.0, L.GQ2, 0.0, L.Q2GHGQ2)

        F = bunchkaufman!(Symmetric(L.Q2GHGQ2, :U), true, check=false)
        if !issuccess(F)
            println("linear system matrix factorization failed")
            BLAS.syrk!('U', 'T', 1.0, L.GQ2, 0.0, L.Q2GHGQ2)
            L.Q2GHGQ2 += 1e-6I
            F = bunchkaufman!(Symmetric(L.Q2GHGQ2, :U), true, check=false)
            if !issuccess(F)
                error("could not fix failure of positive definiteness; terminating")
            end
        end
        ldiv!(F, Q2div)
    end

    BLAS.gemv!('N', 1.0, L.Q2, Q2div, 1.0, Qx)

    GQx = L.G * Qx
    rhs_ty .= L.RiQ1 * (rhs_tx - L.G' * GQx)

    @. rhs_tx = Qx

    @. rhs_tz += GQx

    return
end


# solve 6x6 system for x, y, z, s, kap, tau by solving two symmetric systems and combining the solutions
# TODO update math description
# TODO use in-place mul-add when available in Julia, see https://github.com/JuliaLang/julia/issues/23919
function solvelinsys6!(
    rhs_tx::Vector{Float64},
    rhs_ty::Vector{Float64},
    rhs_tz::Vector{Float64},
    rhs_kap::Float64,
    rhs_ts::Vector{Float64},
    rhs_tau::Float64,
    mu::Float64,
    tau::Float64,
    L::QRChol,
    )
    (zi, yi, xi) = (L.zi, L.yi, L.xi)
    @. yi[:,1] = L.b
    @. yi[:,2] = -rhs_ty
    @. xi[:,1] = -L.c
    @. xi[:,2] = rhs_tx
    z1 = view(zi, :, 1)
    z2 = view(zi, :, 2)

    # calculate z2
    @. z2 = -rhs_tz
    for k in eachindex(L.cone.prmtvs)
        a1k = view(z1, L.cone.idxs[k])
        a2k = view(z2, L.cone.idxs[k])
        a3k = view(rhs_ts, L.cone.idxs[k])
        if L.cone.prmtvs[k].usedual
            @. a1k = a2k - a3k
            calcHiarr_prmtv!(a2k, a1k, L.cone.prmtvs[k])
            a2k ./= mu
        elseif !iszero(a3k) # TODO rhs_ts = 0 for correction steps, so can just check if doing correction
            calcHarr_prmtv!(a1k, a3k, L.cone.prmtvs[k])
            @. a2k -= mu*a1k
        end
    end

    # calculate z1
    if iszero(L.h) # TODO can check once when creating cache
        z1 .= 0.0
    else
        for k in eachindex(L.cone.prmtvs)
            a1k = view(L.h, L.cone.idxs[k])
            a2k = view(z1, L.cone.idxs[k])
            if L.cone.prmtvs[k].usedual
                calcHiarr_prmtv!(a2k, a1k, L.cone.prmtvs[k])
                a2k ./= mu
            else
                calcHarr_prmtv!(a2k, a1k, L.cone.prmtvs[k])
                a2k .*= mu
            end
        end
    end

    # bxGHbz = bx + G'*Hbz
    mul!(L.bxGHbz, L.G', zi)
    @. L.bxGHbz += xi

    # Q1x = Q1*Ri'*by
    mul!(L.Q1x, L.RiQ1', yi)

    # Q2x = Q2*(K22_F\(Q2'*(bxGHbz - GHG*Q1x)))
    mul!(L.GQ1x, L.G, L.Q1x)
    for k in eachindex(L.cone.prmtvs)
        a1k = view(L.GQ1x, L.cone.idxs[k], :)
        a2k = view(L.HGQ1x, L.cone.idxs[k], :)
        if L.cone.prmtvs[k].usedual
            calcHiarr_prmtv!(a2k, a1k, L.cone.prmtvs[k])
            a2k ./= mu
        else
            calcHarr_prmtv!(a2k, a1k, L.cone.prmtvs[k])
            a2k .*= mu
        end
    end
    mul!(L.GHGQ1x, L.G', L.HGQ1x)
    @. L.GHGQ1x = L.bxGHbz - L.GHGQ1x
    mul!(L.Q2div, L.Q2', L.GHGQ1x)

    if size(L.Q2div, 1) > 0
        for k in eachindex(L.cone.prmtvs)
            a1k = view(L.GQ2, L.cone.idxs[k], :)
            a2k = view(L.HGQ2, L.cone.idxs[k], :)
            if L.cone.prmtvs[k].usedual
                calcHiarr_prmtv!(a2k, a1k, L.cone.prmtvs[k])
                a2k ./= mu
            else
                calcHarr_prmtv!(a2k, a1k, L.cone.prmtvs[k])
                a2k .*= mu
            end
        end
        mul!(L.Q2GHGQ2, L.GQ2', L.HGQ2)

        # F = bunchkaufman!(Symmetric(L.Q2GHGQ2), true, check=false)
        # if !issuccess(F)
        #     println("linear system matrix factorization failed")
        #     mul!(L.Q2GHGQ2, L.GQ2', L.HGQ2)
        #     L.Q2GHGQ2 += 1e-6I
        #     F = bunchkaufman!(Symmetric(L.Q2GHGQ2), true, check=false)
        #     if !issuccess(F)
        #         error("could not fix failure of positive definiteness; terminating")
        #     end
        # end
        # ldiv!(F, L.Q2div)

        success = hypatia_sysvx!(L.Q2divcopy, L.Q2GHGQ2, L.Q2div, L.lsferr, L.lsberr, L.lswork, L.lsiwork, L.lsAF, L.ipiv)
        if !success
            # println("linear system matrix factorization failed")
            mul!(L.Q2GHGQ2, L.GQ2', L.HGQ2)
            L.Q2GHGQ2 += 1e-4I
            mul!(L.Q2div, L.Q2', L.GHGQ1x)
            success = hypatia_sysvx!(L.Q2divcopy, L.Q2GHGQ2, L.Q2div, L.lsferr, L.lsberr, L.lswork, L.lsiwork, L.lsAF, L.ipiv)
            if !success
                error("could not fix linear system solve failure; terminating")
            end
        end
        L.Q2div .= L.Q2divcopy
    end
    mul!(L.Q2x, L.Q2, L.Q2div)

    # xi = Q1x + Q2x
    @. xi = L.Q1x + L.Q2x

    # yi = Ri*Q1'*(bxGHbz - GHG*xi)
    mul!(L.Gxi, L.G, xi)
    for k in eachindex(L.cone.prmtvs)
        a1k = view(L.Gxi, L.cone.idxs[k], :)
        a2k = view(L.HGxi, L.cone.idxs[k], :)
        if L.cone.prmtvs[k].usedual
            calcHiarr_prmtv!(a2k, a1k, L.cone.prmtvs[k])
            a2k ./= mu
        else
            calcHarr_prmtv!(a2k, a1k, L.cone.prmtvs[k])
            a2k .*= mu
        end
    end
    mul!(L.GHGxi, L.G', L.HGxi)
    @. L.bxGHbz -= L.GHGxi
    mul!(yi, L.RiQ1, L.bxGHbz)

    # zi = HG*xi - Hbz
    @. zi = L.HGxi - zi

    # combine
    @views dir_tau = (rhs_tau + rhs_kap + dot(L.c, xi[:,2]) + dot(L.b, yi[:,2]) + dot(L.h, z2))/(mu/tau/tau - dot(L.c, xi[:,1]) - dot(L.b, yi[:,1]) - dot(L.h, z1))
    @. @views rhs_tx = xi[:,2] + dir_tau*xi[:,1]
    @. @views rhs_ty = yi[:,2] + dir_tau*yi[:,1]
    @. rhs_tz = z2 + dir_tau*z1
    mul!(z1, L.G, rhs_tx)
    @. rhs_ts = -z1 + L.h*dir_tau - rhs_ts
    dir_kap = -dot(L.c, rhs_tx) - dot(L.b, rhs_ty) - dot(L.h, rhs_tz) - rhs_tau

    return (dir_kap, dir_tau)
end
