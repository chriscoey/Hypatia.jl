#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

polymin real: formulates and solves the real polynomial optimization problem for a given polynomial; see:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming.

polymin complex: minimizes a real-valued complex polynomial over a domain defined by real-valued complex polynomials

TODO
- generalize ModelUtilities interpolation code for complex polynomials space
- merge real and complex polyvars data
- implement PSD formulation for complex case
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

# real polynomials
function polymin_native(
    T::Type{<:Real},
    interp_vals::Vector{T},
    Ps::Vector{Matrix{T}},
    true_min::Real,
    use_primal::Bool, # solve primal, else solve dual
    use_wsos::Bool, # use wsosinterpnonnegative cone, else PSD formulation
    use_linops::Bool,
    )
    if use_primal && !use_wsos
        error("primal psd formulation is not implemented yet")
    end
    U = length(interp_vals)

    cones = CO.Cone{T}[]
    if use_wsos
        push!(cones, CO.WSOSInterpNonnegative{T, T}(U, Ps, use_dual = !use_primal))
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
        true_min = -true_min
    else
        c = interp_vals
        if use_linops
            A = BlockMatrix{T}(1, U, [ones(T, 1, U)], [1:1], [1:U])
        else
            A = ones(T, 1, U) # NOTE can eliminate constraint and a variable
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
            for Pk in Ps
                Lk = size(Pk, 2)
                dk = CO.svec_length(Lk)
                push!(cones, CO.PosSemidefTri{T, T}(dk))
                Gk = Matrix{T}(undef, dk, U)
                l = 1
                for i in 1:Lk, j in 1:i
                    @. Gk[l, :] = -Pk[:, i] * Pk[:, j]
                    l += 1
                end
                MU.vec_to_svec!(Gk, rt2 = sqrt(T(2)))
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

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, true_min = true_min)
end

polymin_native(
    T::Type{<:Real},
    poly_name::Symbol,
    halfdeg::Int,
    args...
    ) = polymin_native(T, get_interp_data(T, poly_name, halfdeg)..., args...)

polymin_native(
    T::Type{<:Real},
    n::Int,
    halfdeg::Int,
    args...
    ) = polymin_native(T, random_interp_data(T, n, halfdeg)..., args...)

# real-valued complex polynomials
function polymin_native(
    T::Type{<:Real},
    ::Type{Complex},
    poly_name::Symbol,
    halfdeg::Int,
    use_primal::Bool,
    use_wsos::Bool;
    sample_factor::Int = 100,
    use_QR::Bool = false,
    )
    if !use_wsos
        error("PSD formulation is not implemented yet")
    end

    (n, f, gs, g_halfdegs, true_min) = complex_poly_data[poly_name]

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
        true_min = -true_min
    else
        c = f.(points)
        A = ones(T, 1, U) # NOTE can eliminate equality and a variable
        b = T[1]
        G = Diagonal(-one(T) * I, U)
        h = zeros(T, U)
    end
    cones = CO.Cone{T}[CO.WSOSInterpNonnegative{T, Complex{T}}(U, P_data, use_dual = !use_primal)]

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, true_min = true_min)
end

function test_polymin_native(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = polymin_native(T, instance...)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @test r.status == :Optimal
    if r.status == :Optimal && !isnan(d.true_min)
        @test r.primal_obj ≈ d.true_min atol = 1e-4 rtol = 1e-4
    end
    return r
end

polymin_native_fast = [
    (:butcher, 2, true, true, false),
    (:caprasse, 4, true, true, false),
    (:goldsteinprice, 7, true, true, false),
    (:goldsteinprice_ball, 7, true, true, false),
    (:goldsteinprice_ellipsoid, 7, true, true, false),
    (:heart, 2, true, true, false),
    (:lotkavolterra, 3, true, true, false),
    (:magnetism7, 2, true, true, false),
    (:magnetism7_ball, 2, true, true, false),
    (:motzkin, 3, true, true, false),
    (:motzkin_ball, 3, true, true, false),
    (:motzkin_ellipsoid, 3, true, true, false),
    (:reactiondiffusion, 4, true, true, false),
    (:robinson, 8, true, true, false),
    (:robinson_ball, 8, true, true, false),
    (:rosenbrock, 5, true, true, false),
    (:rosenbrock_ball, 5, true, true, false),
    (:schwefel, 2, true, true, false),
    (:schwefel_ball, 2, true, true, false),
    (:lotkavolterra, 3, false, true, false),
    (:motzkin, 3, false, true, false),
    (:motzkin_ball, 3, false, true, false),
    (:schwefel, 2, false, true, false),
    (:lotkavolterra, 3, false, false, false),
    (:motzkin, 3, false, false, false),
    (:motzkin_ball, 3, false, false, false),
    (:schwefel, 2, false, false, false),
    (1, 8, true, true, false),
    (2, 5, true, true, false),
    (3, 3, true, true, false),
    (5, 2, true, true, false),
    (3, 3, false, true, false),
    (3, 3, false, false, false),
    (Complex, :abs1d, 1, true, true),
    (Complex, :abs1d, 3, true, true),
    (Complex, :absunit1d, 1, true, true),
    (Complex, :absunit1d, 3, true, true),
    (Complex, :negabsunit1d, 2, true, true),
    (Complex, :absball2d, 1, true, true),
    (Complex, :absbox2d, 2, true, true),
    (Complex, :negabsbox2d, 1, true, true),
    (Complex, :denseunit1d, 2, true, true),
    (Complex, :abs1d, 1, false, true),
    (Complex, :negabsunit1d, 2, false, true),
    (Complex, :absball2d, 1, false, true),
    (Complex, :negabsbox2d, 1, false, true),
    (Complex, :denseunit1d, 2, false, true),
    # (Complex, :abs1d, 1, false, false),
    # (Complex, :negabsunit1d, 2, false, false),
    # (Complex, :absball2d, 1, false, false),
    # (Complex, :negabsbox2d, 1, false, false),
    # (Complex, :denseunit1d, 2, false, false),
    ]
polymin_native_slow = [
    # TODO
    ]
polymin_native_linops = [
    (:butcher, 2, true, true, true),
    (:caprasse, 4, true, true, true),
    (:goldsteinprice, 7, true, true, true),
    (1, 8, true, true, true),
    (2, 5, true, true, true),
    (3, 3, true, true, true),
    (5, 2, true, true, true),
    (3, 3, false, true, true),
    (3, 3, false, false, true),
    ]
