#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/polyEnv.m
formulates and solves the (dual of the) polynomial envelope problem described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

using Hypatia
using LinearAlgebra
using SparseArrays
using DelimitedFiles
using Random
using Test

function build_envelope(
    npoly::Int,
    deg::Int,
    n::Int,
    d::Int;
    primal_wsos::Bool = true,
    use_data::Bool = false,
    usedense::Bool = false,
    rseed::Int = 1,
    )
    # generate interpolation
    @assert deg <= d
    domain = Hypatia.Box(-ones(n), ones(n))
    (U, pts, P0, PWts, w) = Hypatia.interpolate(domain, d, sample=false, calc_w=true)

    if use_data
        # use provided data in data folder
        c_or_h = vec(readdlm(joinpath(@__DIR__, "data/c$(npoly*U).txt"), ',', Float64))
    else
        # generate random data
        Random.seed!(rseed)
        LDegs = binomial(n+deg, n)
        c_or_h = vec(P0[:, 1:LDegs]*rand(-9:9, LDegs, npoly))
    end

    if primal_wsos
        # use formulation with WSOS cone in primal
        P = spzeros(U, U)
        c = -w
        A = zeros(0, U)
        b = Float64[]
        if usedense
            G = repeat(Array(1.0I, U, U), outer=(npoly, 1))
        else
            G = repeat(sparse(1.0I, U, U), outer=(npoly, 1))
        end
        h = c_or_h
    else
        # use formulation with WSOS cone in dual
        P = spzeros(npoly*U, npoly*U)
        c = c_or_h
        if usedense
            A = repeat(Array(1.0I, U, U), outer=(1, npoly))
        else
            A = repeat(sparse(1.0I, U, U), outer=(1, npoly))
        end
        b = w
        G = Diagonal(-1.0I, npoly*U) # TODO uniformscaling
        h = zeros(npoly*U)
    end

    cone = Hypatia.Cone(
        [Hypatia.WSOSPolyInterp(U, [P0, PWts...], !primal_wsos) for k in 1:npoly],
        [1+(k-1)*U:k*U for k in 1:npoly],
        )

    return (P, c, A, b, G, h, cone)
end

function run_envelope(primal_wsos::Bool, usedense::Bool)
    # optionally use fixed data in folder
    # select number of polynomials and degrees for the envelope
    # select dimension and SOS degree (to be squared)
    (P, c, A, b, G, h, cone) =
        # build_envelope(2, 5, 1, 5, use_data=true, primal_wsos=primal_wsos, usedense=usedense)
        # build_envelope(2, 5, 2, 6, primal_wsos=primal_wsos, usedense=usedense)
        # build_envelope(3, 5, 3, 5, primal_wsos=primal_wsos, usedense=usedense)
        build_envelope(2, 30, 1, 30, primal_wsos=primal_wsos, usedense=usedense)

    Hypatia.check_data(P, c, A, b, G, h, cone)

    # (P1, c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(P, c, A, b, G, useQR=false)
    # L = Hypatia.QRSymmCache(c1, A1, b1, G1, h, cone, Q2, RiQ1)
    # L = Hypatia.NaiveCache(P1, c1, A1, b1, G1, h, cone)
    L = Hypatia.Naive3Cache(P, c, A, b, G, h, cone)

    mdl = Hypatia.Model(maxiter=500, verbose=true)
    # Hypatia.load_data!(mdl, P1, c1, A1, b1, G1, h, cone, L)
    Hypatia.load_data!(mdl, P, c, A, b, G, h, cone, L)
    Hypatia.solve!(mdl)

    # x = zeros(length(c))
    # x[dukeep] = Hypatia.get_x(mdl)
    # y = zeros(length(b))
    # y[prkeep] = Hypatia.get_y(mdl)
    # s = Hypatia.get_s(mdl)
    # z = Hypatia.get_z(mdl)
    #
    # status = Hypatia.get_status(mdl)
    # solvetime = Hypatia.get_solvetime(mdl)
    # pobj = Hypatia.get_pobj(mdl)
    # dobj = Hypatia.get_dobj(mdl)
    #
    # @test status == :Optimal
    # @show status
    # @show x
    # @show pobj
    # @show dobj
    return nothing
end

run_envelope_primal_dense() = run_envelope(true, true)
run_envelope_dual_dense() = run_envelope(false, true)
run_envelope_primal_sparse() = run_envelope(true, false)
run_envelope_dual_sparse() = run_envelope(false, false)
