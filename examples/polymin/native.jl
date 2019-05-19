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
const HYP = Hypatia
const CO = HYP.Cones
const MO = HYP.Models
const SO = HYP.Solvers

using Printf

include(joinpath(@__DIR__, "data.jl"))

function polyminreal(
    polyname::Symbol,
    halfdeg::Int;
    primal_wsos::Bool = true,
    )
    (x, fn, dom, true_obj) = getpolydata(polyname)
    sample = (length(x) >= 5) || !isa(dom, MU.Box)
    interp_time = @elapsed begin
        (U, pts, P0, PWts, _) = MU.interpolate(dom, halfdeg, sample = sample)
    end

    # set up problem data
    interp_vals = [fn(pts[j, :]...) for j in 1:U]
    if primal_wsos
        c = [-1.0]
        A = zeros(0, 1)
        b = Float64[]
        G = ones(U, 1)
        h = interp_vals
        true_obj *= -1
    else
        c = interp_vals
        A = ones(1, U) # TODO eliminate constraint and first variable
        b = [1.0]
        G = Diagonal(-1.0I, U) # TODO use UniformScaling
        h = zeros(U)
    end
    cones = [CO.WSOSPolyInterp(U, [P0, PWts...], !primal_wsos)]
    cone_idxs = [1:U]

    nu = sum(size(p, 2) for p in PWts) + size(P0, 2)

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs,
        true_obj = true_obj, n = length(x), halfdeg = halfdeg, nu = nu, interp_time = interp_time)
end

polyminreal1() = polyminreal(:heart, 2)
polyminreal2() = polyminreal(:schwefel, 2)
polyminreal3() = polyminreal(:magnetism7_ball, 2)
polyminreal4() = polyminreal(:motzkin_ellipsoid, 4)
polyminreal5() = polyminreal(:caprasse, 4)
polyminreal6() = polyminreal(:goldsteinprice, 7)
polyminreal7() = polyminreal(:lotkavolterra, 3)
polyminreal8() = polyminreal(:robinson, 8)
polyminreal9() = polyminreal(:robinson_ball, 8)
polyminreal10() = polyminreal(:rosenbrock, 5)
polyminreal11() = polyminreal(:butcher, 2)
polyminreal12() = polyminreal(:goldsteinprice_ellipsoid, 7)
polyminreal13() = polyminreal(:goldsteinprice_ball, 7)

polyminreal14() = polyminreal(:motzkin, 3, primal_wsos = false)
polyminreal15() = polyminreal(:motzkin, 3)
polyminreal16() = polyminreal(:reactiondiffusion, 4, primal_wsos = false)
polyminreal17() = polyminreal(:lotkavolterra, 3, primal_wsos = false)



polyminreal18() = polyminreal(:heart, 2, primal_wsos = false)
polyminreal19() = polyminreal(:schwefel, 2, primal_wsos = false)
polyminreal20() = polyminreal(:magnetism7_ball, 2, primal_wsos = false)
polyminreal21() = polyminreal(:motzkin_ellipsoid, 4, primal_wsos = false)
polyminreal22() = polyminreal(:caprasse, 4, primal_wsos = false)
polyminreal23() = polyminreal(:goldsteinprice, 7, primal_wsos = false)
polyminreal24() = polyminreal(:robinson, 8, primal_wsos = false)
polyminreal25() = polyminreal(:robinson_ball, 8, primal_wsos = false)
polyminreal26() = polyminreal(:rosenbrock, 5, primal_wsos = false)
polyminreal27() = polyminreal(:butcher, 2, primal_wsos = false)
polyminreal28() = polyminreal(:goldsteinprice_ellipsoid, 7, primal_wsos = false)
polyminreal29() = polyminreal(:goldsteinprice_ball, 7, primal_wsos = false)

polyminreal30() = polyminreal(:heart, 3, primal_wsos = false)
polyminreal31() = polyminreal(:schwefel, 3, primal_wsos = false)
polyminreal32() = polyminreal(:magnetism7_ball, 3, primal_wsos = false)
polyminreal33() = polyminreal(:motzkin_ellipsoid, 5, primal_wsos = false)
polyminreal34() = polyminreal(:caprasse, 5, primal_wsos = false)
polyminreal35() = polyminreal(:goldsteinprice, 8, primal_wsos = false)
polyminreal36() = polyminreal(:robinson, 9, primal_wsos = false)
polyminreal37() = polyminreal(:robinson_ball, 9, primal_wsos = false)
polyminreal38() = polyminreal(:rosenbrock, 6, primal_wsos = false)
polyminreal39() = polyminreal(:butcher, 3, primal_wsos = false)
polyminreal40() = polyminreal(:goldsteinprice_ellipsoid, 8, primal_wsos = false)
polyminreal41() = polyminreal(:goldsteinprice_ball, 8, primal_wsos = false)


# polyminreal33() = polyminreal(:heart, 2, primal_wsos = false)
# polyminreal34() = polyminreal(:schwefel, 2, primal_wsos = false)
# polyminreal35() = polyminreal(:magnetism7_ball, 2, primal_wsos = false)
# polyminreal36() = polyminreal(:motzkin_ellipsoid, 4, primal_wsos = false)
# polyminreal37() = polyminreal(:caprasse, 4, primal_wsos = false)
# polyminreal38() = polyminreal(:goldsteinprice, 7, primal_wsos = false)
# polyminreal39() = polyminreal(:robinson, 8, primal_wsos = false)
# polyminreal40() = polyminreal(:robinson_ball, 8, primal_wsos = false)
# polyminreal41() = polyminreal(:rosenbrock, 5, primal_wsos = false)
# polyminreal42() = polyminreal(:butcher, 2, primal_wsos = false)
# polyminreal43() = polyminreal(:goldsteinprice_ellipsoid, 7, primal_wsos = false)
# polyminreal44() = polyminreal(:goldsteinprice_ball, 7, primal_wsos = false)

function polymincomplex(
    polyname::Symbol,
    halfdeg::Int;
    primal_wsos = true,
    sample_factor::Int = 100,
    use_QR::Bool = false,
    )
    (n, deg, f, gs, g_halfdegs, true_obj) = complexpolys[polyname]

    # generate interpolation
    # TODO use more numerically-stable basis for columns
    L = binomial(n + halfdeg, n)
    U = L^2
    L_basis = [a for t in 0:halfdeg for a in Combinatorics.multiexponents(n, t)]
    mon_pow(z, ex) = prod(z[i]^ex[i] for i in eachindex(ex))
    V_basis = [z -> mon_pow(z, L_basis[k]) * mon_pow(conj(z), L_basis[l]) for l in eachindex(L_basis) for k in eachindex(L_basis)]
    @assert length(V_basis) == U

    # sample from domain (inefficient for general domains, only samples from unit box and checks feasibility)
    interp_time = @elapsed begin
    num_samples = sample_factor * U
    samples = Vector{Vector{ComplexF64}}(undef, num_samples)
    k = 0
    randbox() = 2 * rand() - 1
    while k < num_samples
        z = [Complex(randbox(), randbox()) for i in 1:n]
        if all(g -> g(z) > 0.0, gs)
            k += 1
            samples[k] = z
        end
    end

    # select subset of points to maximize |det(V)| in heuristic QR-based procedure (analogous to real case)
    V = [b(z) for z in samples, b in V_basis]
    @test rank(V) == U
    VF = qr(Matrix(transpose(V)), Val(true))
    keep = VF.p[1:U]
    points = samples[keep]
    V = V[keep, :]
    @test rank(V) == U
    end # timing interpolation related things

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

    nu = sum(size(p, 2) for p in P_data)

    # setup problem data
    if primal_wsos
        c = [-1.0]
        A = zeros(0, 1)
        b = Float64[]
        G = ones(U, 1)
        h = f.(points)
        true_obj *= -1
    else
        c = f.(points)
        A = ones(1, U) # TODO can eliminate equality and a variable
        b = [1.0]
        G = Diagonal(-1.0I, U)
        h = zeros(U)
    end
    cones = [CO.WSOSPolyInterp(U, P_data, !primal_wsos)]
    cone_idxs = [1:U]

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs,
        true_obj = true_obj, n = n, halfdeg = halfdeg, interp_time = interp_time, nu = nu)
end

polymincomplex1() = polymincomplex(:abs1d, 1)
polymincomplex2() = polymincomplex(:absunit1d, 1)
polymincomplex3() = polymincomplex(:negabsunit1d, 2)
polymincomplex4() = polymincomplex(:absball2d, 1)
polymincomplex5() = polymincomplex(:absbox2d, 2)
polymincomplex6() = polymincomplex(:negabsbox2d, 1)
polymincomplex7() = polymincomplex(:denseunit1d, 2)
polymincomplex8() = polymincomplex(:abs1d, 1, primal_wsos = false)
polymincomplex9() = polymincomplex(:absunit1d, 1, primal_wsos = false)
polymincomplex10() = polymincomplex(:negabsunit1d, 2, primal_wsos = false)
polymincomplex11() = polymincomplex(:absball2d, 1, primal_wsos = false)
polymincomplex12() = polymincomplex(:absbox2d, 2, primal_wsos = false)
polymincomplex13() = polymincomplex(:negabsbox2d, 1, primal_wsos = false)
polymincomplex14() = polymincomplex(:denseunit1d, 2, primal_wsos = false)

polymincomplex15() = polymincomplex(:abs1d, 2, primal_wsos = false)
polymincomplex16() = polymincomplex(:absunit1d, 2, primal_wsos = false)
polymincomplex17() = polymincomplex(:negabsunit1d, 3, primal_wsos = false)
polymincomplex18() = polymincomplex(:absball2d, 2, primal_wsos = false)
polymincomplex19() = polymincomplex(:absbox2d, 3, primal_wsos = false)
polymincomplex20() = polymincomplex(:negabsbox2d, 2, primal_wsos = false)
polymincomplex21() = polymincomplex(:denseunit1d, 3, primal_wsos = false)

function test_polymin(instance::Function; options, rseed::Int = 1, cumulative::Bool = false)
    Random.seed!(rseed)
    if !cumulative
        reset_timer!(Hypatia.to)
        repeats = 1
    else
        repeats = 0
    end
    d = instance()

    for nbhd in ["_infty", "_hess"]

        infty_nbhd = (nbhd == "_infty")

        model = MO.PreprocessedLinearModel(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
        stepper = SO.CombinedHSDStepper(model, infty_nbhd = infty_nbhd)
        solver = SO.HSDSolver(model; options..., stepper = stepper)
        SO.solve(solver)
        build_time = 0

        for _ in 1:repeats
            reset_timer!(Hypatia.to)
            build_time = @elapsed model = MO.PreprocessedLinearModel(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
            stepper = SO.CombinedHSDStepper(model, infty_nbhd = infty_nbhd)
            solver = SO.HSDSolver(model; options..., stepper = stepper)
            SO.solve(solver)
        end
        r = SO.get_certificates(solver, model, test = true, atol = 1e-4, rtol = 1e-4)
        @test r.status == :Optimal
        @test r.primal_obj ≈ d.true_obj atol = 1e-4 rtol = 1e-4

        if !cumulative
            polyname = string(methods(instance).mt.name)
            open(joinpath("timings", polyname * nbhd * ".txt"), "w") do f
                print_timer(f, Hypatia.to) # methods(instance).mt.name
            end

            open(joinpath("timings", "allpolymins" * nbhd * ".csv"), "a") do f
                G1 = size(d.G, 1)
                tt = TimerOutputs.tottime(Hypatia.to) # total solving time (nanoseconds)
                tts = tt / 1e6
                tb = build_time
                ta = TimerOutputs.time(Hypatia.to["aff alpha"]) / tt # % of time in affine alpha
                tc = TimerOutputs.time(Hypatia.to["comb alpha"]) / tt # % of time in comb alpha
                td = TimerOutputs.time(Hypatia.to["directions"]) / tt # % of time calculating directions
                ti = d.interp_time
                num_iters = TimerOutputs.ncalls(Hypatia.to["directions"])
                aff_per_iter = TimerOutputs.ncalls(Hypatia.to["aff alpha"]["linstep"]) / num_iters
                comb_per_iter = TimerOutputs.ncalls(Hypatia.to["comb alpha"]["linstep"]) / num_iters
                # println(f, "$polyname, $(d.n), $(d.halfdeg), $G1, $G2, $tt, $ta, $ti, $num_iters, $aff_per_iter, $comb_per_iter")

                @printf(f, "%15s, %15d, %15d, %15d, %15d, %15.2f, %15.2f, %15.2f, %15.2f, %15d, %15.2f, %15.2f, %15.2f\n",
                    polyname, d.n, d.halfdeg, G1, d.nu, tts, tb, ta, ti, num_iters, aff_per_iter, comb_per_iter, td
                    )
            end
        end
    end # nbhd


    return
end

test_polymin_dual_hearts(; options...) = test_polymin.([
    polyminreal30,
    ], options = options)

test_polymin_duals(; options...) = test_polymin.([
    polyminreal14,
    polyminreal16,
    polyminreal17,
    polyminreal18,
    polyminreal19,
    polyminreal20,
    polyminreal21,
    polyminreal22,
    polyminreal23,
    polyminreal24,
    polyminreal25,
    polyminreal26,
    polyminreal27,
    polyminreal28,
    polyminreal29,
    polymincomplex8,
    polymincomplex9,
    polymincomplex10,
    polymincomplex11,
    polymincomplex12,
    polymincomplex13,
    polymincomplex14,
    # polyminreal30, # extra heart
    ], options = options)

test_polymin_all(; options...) = test_polymin.([
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
    ], options = options)

test_polymin(; options...) = test_polymin.([
    polyminreal2,
    polyminreal3,
    polyminreal12,
    polyminreal14,
    polymincomplex7,
    polymincomplex14,
    ], options = options)
