#=
naive method that simply performs one high-dimensional linear system solve
TODO currently only does dense operations, needs to work for sparse
=#
mutable struct NaiveCache <: LinSysCache
    c
    A
    b
    G
    h
    LHS3
    LHS3copy
    rhs3
    LHS6
    LHS6copy
    rhs6
    tyk
    tzk
    tkk
    tsk
    ttk

    function NaiveCache(
        c::Vector{Float64},
        A::AbstractMatrix{Float64},
        b::Vector{Float64},
        G::AbstractMatrix{Float64},
        h::Vector{Float64},
        )

        (n, p, q) = (length(c), length(b), length(h))
        L = new()
        L.c = c
        L.A = A
        L.b = b
        L.G = G
        L.h = h
        L.tyk = n+1
        L.tzk = L.tyk + p
        L.tkk = L.tzk + q
        L.tsk = L.tkk + 1
        L.ttk = L.tsk + q
        # tx ty tz
        L.LHS3 = [
            zeros(n,n)  A'          G';
            A           zeros(p,p)  zeros(p,q);
            G           zeros(q,p)  Matrix(-1.0I,q,q);
            ]
        L.LHS3copy = similar(L.LHS3)
        L.rhs3 = zeros(L.tkk-1)
        # tx ty tz kap ts tau
        L.LHS6 = [
            zeros(n,n)  A'          G'                zeros(n)  zeros(n,q)         c;
            -A          zeros(p,p)  zeros(p,q)        zeros(p)  zeros(p,q)         b;
            zeros(q,n)  zeros(q,p)  Matrix(1.0I,q,q)  zeros(q)  Matrix(1.0I,q,q)   zeros(q);
            zeros(1,n)  zeros(1,p)  zeros(1,q)        1.0       zeros(1,q)         1.0;
            -G          zeros(q,p)  zeros(q,q)        zeros(q)  Matrix(-1.0I,q,q)  h;
            -c'         -b'         -h'               -1.0      zeros(1,q)         0.0;
            ]
        L.LHS6copy = similar(L.LHS6)
        L.rhs6 = zeros(L.ttk)

        return L
    end
end

# solve system for x, y, z
function solvelinsys3!(
    rhs_tx::Vector{Float64},
    rhs_ty::Vector{Float64},
    rhs_tz::Vector{Float64},
    H::AbstractMatrix{Float64},
    L::NaiveCache,
    )

    rhs = L.rhs3
    rhs[1:L.tyk-1] = rhs_tx
    @. rhs[L.tyk:L.tzk-1] = -rhs_ty
    @. rhs[L.tzk:L.tkk-1] = -rhs_tz

    @. L.LHS3copy = L.LHS3
    @. L.LHS3copy[L.tzk:L.tkk-1, L.tzk:L.tkk-1] = -H

    F = bunchkaufman!(Symmetric(L.LHS3copy))
    ldiv!(F, rhs)

    @. @views begin
        rhs_tx = rhs[1:L.tyk-1]
        rhs_ty = rhs[L.tyk:L.tzk-1]
        rhs_tz = rhs[L.tzk:L.tkk-1]
    end

    return nothing
end

# solve system for x, y, z, s, kap, tau
function solvelinsys6!(
    rhs_tx::Vector{Float64},
    rhs_ty::Vector{Float64},
    rhs_tz::Vector{Float64},
    rhs_ts::Vector{Float64},
    rhs_kap::Float64,
    rhs_tau::Float64,
    mu::Float64,
    tau::Float64,
    H::AbstractMatrix{Float64},
    L::NaiveCache,
    )

    rhs = L.rhs6
    rhs[1:L.tyk-1] = rhs_tx
    rhs[L.tyk:L.tzk-1] = rhs_ty
    rhs[L.tzk:L.tkk-1] = rhs_tz
    rhs[L.tkk] = rhs_kap
    rhs[L.tsk:L.ttk-1] = rhs_ts
    rhs[end] = rhs_tau

    @. L.LHS6copy = L.LHS6
    L.LHS6copy[L.tzk:L.tkk-1, L.tsk:L.ttk-1] = H
    L.LHS6copy[L.tkk, end] = mu/tau/tau

    F = qr!(L.LHS6copy)
    ldiv!(F, rhs)

    @. @views begin
        rhs_tx = rhs[1:L.tyk-1]
        rhs_ty = rhs[L.tyk:L.tzk-1]
        rhs_tz = rhs[L.tzk:L.tkk-1]
        rhs_ts = rhs[L.tsk:L.ttk-1]
    end
    dir_kap = rhs[L.tkk]
    dir_tau = rhs[end]

    return (dir_kap, dir_tau)
end
