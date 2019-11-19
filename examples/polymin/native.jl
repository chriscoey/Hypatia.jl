#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

polyminreal: formulates and solves the real polynomial optimization problem for a given polynomial; see:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming.

polymincomplex: minimizes a real-valued complex polynomial over a domain defined by real-valued complex polynomials

TODO
- generalize ModelUtilities interpolation code for complex polynomials space
- merge real and complex polyvars data when complex is supported in DynamicPolynomials: https://github.com/JuliaAlgebra/MultivariatePolynomials.jl/issues/11
=#

import Random
using LinearAlgebra
import Combinatorics
using Test
import Hypatia
import Hypatia.BlockMatrix
const CO = Hypatia.Cones
const MU = Hypatia.ModelUtilities

include(joinpath(@__DIR__, "data.jl"))

function polyminreal(
    T::Type{<:Real},
    polyname::Symbol,
    halfdeg::Int;
    use_primal::Bool = true,
    use_wsos::Bool = true,
    n::Int = 0,
    use_linops::Bool = false,
    )
    if use_primal && !use_wsos
        error("primal psd formulation is not implemented yet")
    end

    if polyname == :random
        if n <= 0
            error("`n` should be specified as a positive keyword argument if randomly generating a polynomial")
        end
        true_obj = NaN
        dom = MU.Box{T}(-ones(T, n), ones(T, n))
        (U, pts, P0, PWts, _) = MU.interpolate(dom, halfdeg, sample = true)
        interp_vals = T.(randn(U))
    else
        (x, fn, dom, true_obj) = getpolydata(polyname, T = T)
        sample = (length(x) >= 5) || !isa(dom, MU.Box)
        (U, pts, P0, PWts, _) = MU.interpolate(dom, halfdeg, sample = sample)
        # set up problem data
        interp_vals = T[fn(pts[j, :]...) for j in 1:U]
    end
    @show typeof(P0)
    PWts = convert.(Matrix{T}, PWts)

    if use_wsos
        cones = CO.Cone{T}[CO.WSOSPolyInterp{T, T}(U, [P0, PWts...], !use_primal)]
    else
        # will be set up iteratively
        cones = CO.Cone{T}[]
    end

    if use_primal
        c = T[-1]
        if use_linops
            A = BlockMatrix{T}(0, 1, Any[[]], [0:-1], [1:1])
            G = BlockMatrix{T}(U, 1, [ones(T, U, 1)], [1:U], [1:1])
        else
            A = zeros(T, 0, 1)
            G = ones(T, U, 1)
        end
        b = T[]
        h = interp_vals
        true_obj = -true_obj
    else
        c = interp_vals
        if use_linops
            A = BlockMatrix{T}(1, U, [ones(T, 1, U)], [1:1], [1:U])
        else
            A = ones(T, 1, U) # TODO eliminate constraint and first variable
        end
        b = T[1]
        if use_wsos
            if use_linops
                G = BlockMatrix{T}(U, U, [-I], [1:U], [1:U])
            else
                G = Diagonal(-one(T) * I, U)
            end
            h = zeros(T, U)
        else
            G_full = zeros(T, 0, U)
            for Pk in [P0, PWts...]
                Lk = size(Pk, 2)
                dk = div(Lk * (Lk + 1), 2)
                push!(cones, CO.PosSemidefTri{T, T}(dk))
                Gk = Matrix{T}(undef, dk, U)
                l = 1
                for i in 1:Lk, j in 1:i
                    @. Gk[l, :] = -Pk[:, i] * Pk[:, j]
                    l += 1
                end
                MU.vec_to_svec_cols!(Gk, sqrt(T(2)))
                G_full = vcat(G_full, Gk)
            end
            if use_linops
                (nrows, ncols) = size(G_full)
                G = BlockMatrix{T}(nrows, ncols, [G_full], [1:nrows], [1:ncols])
            else
                G = G_full
            end
            h = zeros(T, size(G, 1))
        end
    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, true_obj = true_obj)
end

polyminreal1(T::Type{<:Real}) = polyminreal(T, :heart, 2)
polyminreal2(T::Type{<:Real}) = polyminreal(T, :schwefel, 2)
polyminreal3(T::Type{<:Real}) = polyminreal(T, :magnetism7_ball, 2)
polyminreal4(T::Type{<:Real}) = polyminreal(T, :motzkin_ellipsoid, 4)
polyminreal5(T::Type{<:Real}) = polyminreal(T, :caprasse, 4)
polyminreal6(T::Type{<:Real}) = polyminreal(T, :goldsteinprice, 7)
polyminreal7(T::Type{<:Real}) = polyminreal(T, :lotkavolterra, 3)
polyminreal8(T::Type{<:Real}) = polyminreal(T, :robinson, 8)
polyminreal9(T::Type{<:Real}) = polyminreal(T, :robinson_ball, 8)
polyminreal10(T::Type{<:Real}) = polyminreal(T, :rosenbrock, 5)
polyminreal11(T::Type{<:Real}) = polyminreal(T, :butcher, 2)
polyminreal12(T::Type{<:Real}) = polyminreal(T, :goldsteinprice_ellipsoid, 7)
polyminreal13(T::Type{<:Real}) = polyminreal(T, :goldsteinprice_ball, 7)
polyminreal14(T::Type{<:Real}) = polyminreal(T, :motzkin, 3, use_primal = false)
polyminreal15(T::Type{<:Real}) = polyminreal(T, :motzkin, 3)
polyminreal16(T::Type{<:Real}) = polyminreal(T, :reactiondiffusion, 4, use_primal = false)
polyminreal17(T::Type{<:Real}) = polyminreal(T, :lotkavolterra, 3, use_primal = false)
polyminreal18(T::Type{<:Real}) = polyminreal(T, :motzkin, 3, use_primal = false, use_wsos = false)
polyminreal19(T::Type{<:Real}) = polyminreal(T, :motzkin, 3, use_primal = false, use_wsos = false)
polyminreal20(T::Type{<:Real}) = polyminreal(T, :reactiondiffusion, 4, use_primal = false, use_wsos = false)
polyminreal21(T::Type{<:Real}) = polyminreal(T, :lotkavolterra, 3, use_primal = false, use_wsos = false)
polyminreal22(T::Type{<:Real}) = polyminreal(T, :random, 2, use_primal = true, use_wsos = true, n = 5, use_linops = true)
polyminreal23(T::Type{<:Real}) = polyminreal(T, :random, 2, use_primal = true, use_wsos = true, n = 5, use_linops = false)
polyminreal24(T::Type{<:Real}) = polyminreal(T, :random, 2, use_primal = false, use_wsos = true, n = 5, use_linops = true)
polyminreal25(T::Type{<:Real}) = polyminreal(T, :random, 2, use_primal = false, use_wsos = false, n = 5, use_linops = false)
polyminreal26(T::Type{<:Real}) = polyminreal(T, :random, 2, use_primal = false, use_wsos = true, n = 5, use_linops = true)
polyminreal27(T::Type{<:Real}) = polyminreal(T, :random, 2, use_primal = false, use_wsos = false, n = 5, use_linops = false)

function polymincomplex(
    T::Type{<:Real},
    polyname::Symbol,
    halfdeg::Int;
    use_primal::Bool = true,
    use_wsos::Bool = true,
    sample_factor::Int = 100,
    use_QR::Bool = false,
    )
    if !use_wsos
        error("PSD formulation is not implemented yet")
    end

    (n, f, gs, g_halfdegs, true_obj) = complexpolys[polyname]

    # generate interpolation
    # TODO use more numerically-stable basis for columns
    L = binomial(n + halfdeg, n)
    U = L^2
    L_basis = [a for t in 0:halfdeg for a in Combinatorics.multiexponents(n, t)]
    mon_pow(z, ex) = prod(z[i]^ex[i] for i in eachindex(ex))
    V_basis = [z -> mon_pow(z, L_basis[k]) * mon_pow(conj(z), L_basis[l]) for l in eachindex(L_basis) for k in eachindex(L_basis)]
    @assert length(V_basis) == U

    # sample from domain (inefficient for general domains, only samples from unit box and checks feasibility)
    num_samples = sample_factor * U
    samples = Vector{Vector{Complex{T}}}(undef, num_samples)
    k = 0
    randbox() = 2 * rand(T) - 1
    while k < num_samples
        z = [Complex(randbox(), randbox()) for i in 1:n]
        if all(g -> g(z) > zero(T), gs)
            k += 1
            samples[k] = z
        end
    end

    # select subset of points to maximize |det(V)| in heuristic QR-based procedure (analogous to real case)
    V = [b(z) for z in samples, b in V_basis]
    VF = qr(Matrix(transpose(V)), Val(true))
    keep = VF.p[1:U]
    points = samples[keep]
    V = V[keep, :]

    # setup P matrices
    P0 = V[:, 1:L]
    if use_QR
        P0 = Matrix(qr(P0).Q)
    end
    P_data = [P0]
    for i in eachindex(gs)
        gi = gs[i].(points)
        Pi = Diagonal(sqrt.(gi)) * P0[:, 1:binomial(n + halfdeg - g_halfdegs[i], n)]
        if use_QR
            Pi = Matrix(qr(Pi).Q)
        end
        push!(P_data, Pi)
    end

    # setup problem data
    if use_primal
        c = T[-1]
        A = zeros(T, 0, 1)
        b = T[]
        G = ones(T, U, 1)
        h = f.(points)
        true_obj = -true_obj
    else
        c = f.(points)
        A = ones(T, 1, U) # TODO can eliminate equality and a variable
        b = T[1]
        G = Diagonal(-one(T) * I, U)
        h = zeros(T, U)
    end
    cones = CO.Cone{T}[CO.WSOSPolyInterp{T, Complex{T}}(U, P_data, !use_primal)]

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, true_obj = true_obj)
end

polymincomplex1(T::Type{<:Real}) = polymincomplex(T, :abs1d, 1)
polymincomplex2(T::Type{<:Real}) = polymincomplex(T, :absunit1d, 1)
polymincomplex3(T::Type{<:Real}) = polymincomplex(T, :negabsunit1d, 2)
polymincomplex4(T::Type{<:Real}) = polymincomplex(T, :absball2d, 1)
polymincomplex5(T::Type{<:Real}) = polymincomplex(T, :absbox2d, 2)
polymincomplex6(T::Type{<:Real}) = polymincomplex(T, :negabsbox2d, 1)
polymincomplex7(T::Type{<:Real}) = polymincomplex(T, :denseunit1d, 2)
polymincomplex8(T::Type{<:Real}) = polymincomplex(T, :abs1d, 1, use_primal = false)
polymincomplex9(T::Type{<:Real}) = polymincomplex(T, :absunit1d, 1, use_primal = false)
polymincomplex10(T::Type{<:Real}) = polymincomplex(T, :negabsunit1d, 2, use_primal = false)
polymincomplex11(T::Type{<:Real}) = polymincomplex(T, :absball2d, 1, use_primal = false)
polymincomplex12(T::Type{<:Real}) = polymincomplex(T, :absbox2d, 2, use_primal = false)
polymincomplex13(T::Type{<:Real}) = polymincomplex(T, :negabsbox2d, 1, use_primal = false)
polymincomplex14(T::Type{<:Real}) = polymincomplex(T, :denseunit1d, 2, use_primal = false)

instances_polymin_all = [
    polyminreal1,
    polyminreal2,
    polyminreal3,
    polyminreal4,
    polyminreal5,
    polyminreal6,
    polyminreal7,
    polyminreal8,
    polyminreal9,
    polyminreal10,
    polyminreal11,
    polyminreal12,
    polyminreal13,
    polyminreal14,
    polyminreal15,
    polyminreal16,
    polyminreal17,
    polyminreal18,
    polyminreal19,
    polyminreal20,
    polyminreal21,
    polyminreal23,
    polyminreal25,
    polyminreal27,
    polymincomplex1,
    polymincomplex2,
    polymincomplex3,
    polymincomplex4,
    polymincomplex5,
    polymincomplex6,
    polymincomplex7,
    polymincomplex8,
    polymincomplex9,
    polymincomplex10,
    polymincomplex11,
    polymincomplex12,
    polymincomplex13,
    polymincomplex14,
    ]
instances_polymin_linops = [
    polyminreal22,
    polyminreal24,
    polyminreal26,
    ]
instances_polymin_few = [
    polyminreal2,
    polyminreal3,
    polyminreal12,
    polyminreal14,
    polyminreal18,
    polyminreal23,
    polymincomplex7,
    polymincomplex14,
    ]

function test_polymin(instance::Function; T::Type{<:Real} = Float64, options::NamedTuple = (atol = sqrt(sqrt(eps(T))),), rseed::Int = 1)
    Random.seed!(rseed)
    d = instance(T)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @test r.status == :Optimal
    if !isnan(d.true_obj)
        @test r.primal_obj ≈ d.true_obj atol=options.atol rtol=options.atol
    end
    return
end
