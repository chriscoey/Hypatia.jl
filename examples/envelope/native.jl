#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/polyEnv.m
formulates and solves the (dual of the) polynomial envelope problem described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

using LinearAlgebra
using SparseArrays
import Random
using Test
import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MU = HYP.ModelUtilities
const MO = HYP.Models
const SO = HYP.Solvers

function envelope(
    npoly::Int,
    rand_halfdeg::Int,
    n::Int,
    env_halfdeg::Int;
    primal_wsos::Bool = true,
    )
    # generate interpolation
    @assert rand_halfdeg <= env_halfdeg
    domain = MU.Box(-ones(n), ones(n))
    (U, pts, P0, PWts, w) = MU.interpolate(domain, env_halfdeg, sample = false, calc_w = true)

    # generate random data
    L = binomial(n + rand_halfdeg, n)
    c_or_h = vec(P0[:, 1:L] * rand(-9:9, L, npoly))

    if primal_wsos
        # WSOS cone in primal
        c = -w
        A = zeros(0, U)
        b = Float64[]
        G = repeat(sparse(1.0I, U, U), outer = (npoly, 1))
        h = c_or_h
    else
        # WSOS cone in dual
        c = c_or_h
        A = repeat(sparse(1.0I, U, U), outer = (1, npoly))
        b = w
        G = Diagonal(-1.0I, npoly * U) # TODO uniformscaling
        h = zeros(npoly * U)
    end

    cones = [CO.WSOSPolyInterp(U, [P0, PWts...], !primal_wsos) for k in 1:npoly]
    cone_idxs = [(1 + (k - 1) * U):(k * U) for k in 1:npoly]

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

envelope1() = envelope(2, 5, 2, 6)
envelope2() = envelope(3, 5, 3, 5)
envelope3() = envelope(2, 30, 1, 30)
envelope4() = envelope(2, 5, 2, 6, primal_wsos = false)
envelope5() = envelope(3, 5, 3, 5, primal_wsos = false)
envelope6() = envelope(2, 30, 1, 30, primal_wsos = false)

function test_envelope(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    model = MO.PreprocessedLinearModel(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver(model; options...)
    SO.solve(solver)
    r = SO.get_certificates(solver, model, test = true, atol = 1e-4, rtol = 1e-4)
    @test r.status == :Optimal
    return
end

test_envelope_dual(; options...) = test_envelope.([
    envelope4,
    envelope5,
    envelope6,
    ], options = options)

test_envelope_all(; options...) = test_envelope.([
    envelope1,
    envelope2,
    envelope3,
    envelope4,
    envelope5,
    envelope6,
    ], options = options)

test_envelope(; options...) = test_envelope.([
    envelope1,
    envelope5,
    ], options = options)
