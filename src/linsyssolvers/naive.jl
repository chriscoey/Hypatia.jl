#=
Copyright 2018, Chris Coey and contributors

naive method that simply performs one high-dimensional linear system solve
TODO should allow LHS6 to be sparse
=#

mutable struct NaiveCache <: LinSysCache
    P
    c
    A
    b
    G
    h
    cone

    # LHS3
    # LHS3copy
    # rhs3

    LHS6
    LHS6copy
    rhs6

    tyk
    tzk
    tkk
    tsk
    ttk

    function NaiveCache(
        P::AbstractMatrix{Float64},
        c::Vector{Float64},
        A::AbstractMatrix{Float64},
        b::Vector{Float64},
        G::AbstractMatrix{Float64},
        h::Vector{Float64},
        cone::Cone,
        )
        (n, p, q) = (length(c), length(b), length(h))
        L = new()
        L.P = P
        L.c = c
        L.A = A
        L.b = b
        L.G = G
        L.h = h
        L.cone = cone

        L.tyk = n + 1
        L.tzk = L.tyk + p
        L.tkk = L.tzk + q
        L.tsk = L.tkk + 1
        L.ttk = L.tsk + q

        # tx ty tz
        L.LHS3 = [
            P  A'          G';
            A  zeros(p,p)  zeros(p,q);
            G  zeros(q,p)  -1.0I;
            ]
        L.LHS3copy = similar(L.LHS3)
        L.rhs3 = zeros(L.tkk-1)

        return L
    end
end

# solve system for x, y, z
function solvelinsys3!(
    rhs_tx::Vector{Float64},
    rhs_ty::Vector{Float64},
    rhs_tz::Vector{Float64},
    mu::Float64,
    L::NaiveCache;
    identityH::Bool = false,
    )

    rhs = L.rhs3
    rhs[1:L.tyk-1] = rhs_tx
    @. rhs[L.tyk:L.tzk-1] = -rhs_ty
    @. rhs[L.tzk:L.tkk-1] = -rhs_tz

    @. L.LHS3copy = L.LHS3
    @assert identityH
    # TODO update for prim or dual cones
    # if !identityH
    #     for k in eachindex(L.cone.prmtvs)
    #         idxs = L.tzk - 1 .+ L.cone.idxs[k]
    #         dim = dimension(L.cone.prmtvs[k])
    #         calcHiarr_prmtv!(view(L.LHS3copy, idxs, idxs), Matrix(-inv(mu)*I, dim, dim), L.cone.prmtvs[k])
    #     end
    # end

    F = bunchkaufman!(Symmetric(L.LHS3copy))
    ldiv!(F, rhs)

    @. @views begin
        rhs_tx = rhs[1:L.tyk-1]
        rhs_ty = rhs[L.tyk:L.tzk-1]
        rhs_tz = rhs[L.tzk:L.tkk-1]
    end

    return nothing
end

# # solve system for x, y, z, kap, s, tau
# function solvelinsys6!(
#     rhs_tx::Vector{Float64},
#     rhs_ty::Vector{Float64},
#     rhs_tz::Vector{Float64},
#     rhs_kap::Float64,
#     rhs_ts::Vector{Float64},
#     rhs_tau::Float64,
#     mu::Float64,
#     tau::Float64,
#     L::NaiveCache,
#     )
#     rhs = L.rhs6
#     rhs[1:L.tyk-1] = rhs_tx
#     rhs[L.tyk:L.tzk-1] = rhs_ty
#     rhs[L.tzk:L.tkk-1] = rhs_tz
#     rhs[L.tkk] = rhs_kap
#     rhs[L.tsk:L.ttk-1] = rhs_ts
#     rhs[end] = rhs_tau
#
#     @. L.LHS6copy = L.LHS6
#     L.LHS6copy[L.tkk, end] = mu/tau/tau # TODO note in CVXOPT coneprog doc, there is no rescaling by tau, they to kap*dtau + tau*dkap = -rhskap
#     for k in eachindex(L.cone.prmtvs)
#         dim = dimension(L.cone.prmtvs[k])
#         coloffset = (L.cone.prmtvs[k].usedual ? L.tzk : L.tsk)
#         # TODO don't use Matrix(mu*I, dim, dim) because it allocates and is slow
#         calcHarr_prmtv!(view(L.LHS6copy, L.tzk - 1 .+ L.cone.idxs[k], coloffset - 1 .+ L.cone.idxs[k]), Matrix(mu*I, dim, dim), L.cone.prmtvs[k])
#     end
#
#     F = lu!(L.LHS6copy)
#     ldiv!(F, rhs)
#
#     @. @views begin
#         rhs_tx = rhs[1:L.tyk-1]
#         rhs_ty = rhs[L.tyk:L.tzk-1]
#         rhs_tz = rhs[L.tzk:L.tkk-1]
#         rhs_ts = rhs[L.tsk:L.ttk-1]
#     end
#     dir_kap = rhs[L.tkk]
#     dir_tau = rhs[end]
#
#     return (dir_kap, dir_tau)
# end
