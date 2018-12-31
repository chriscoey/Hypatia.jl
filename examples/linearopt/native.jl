#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/random_lp.m
solves a simple linear optimization problem (LP) min c'x s.t. Ax = b, x >= 0
=#

using Hypatia
using SparseArrays
using DelimitedFiles
using Random
using Test

function build_linearopt(
    m::Int,
    n::Int;
    use_data::Bool = false,
    usedense::Bool = false,
    nzfrac::Float64 = 1/sqrt(n),
    tosparse::Bool = false,
    rseed::Int = 1,
    )
    # set up problem data
    if use_data
        # use provided data in data folder
        datapath = joinpath(@__DIR__, "data")
        A = readdlm(joinpath(datapath, "A$(m)x$(n).txt"), ',', Float64)
        b = vec(readdlm(joinpath(datapath, "b$m.txt"), ',', Float64))
        c = vec(readdlm(joinpath(datapath, "c$n.txt"), ',', Float64))
    else
        # generate random data
        Random.seed!(rseed)
        if usedense
            A = rand(-9.0:9.0, m, n)
        else
            A = 10.0.*sprandn(m, n, nzfrac)
        end
        b = A*ones(n)
        c = rand(0.0:9.0, n)
    end
    if tosparse && !issparse(A)
        A = sparse(A)
    end
    G = Diagonal(-1.0I, n) # TODO uniformscaling
    h = zeros(n)

    cone = Hypatia.Cone([Hypatia.Nonnegative(n)], [1:n])

    return (c, A, b, G, h, cone)
end

function run_linearopt()
    # optionally use fixed data in folder
    # select the random matrix size, dense/sparse, sparsity fraction
    (c, A, b, G, h, cone) =
        # build_linearopt(500, 1000, use_data=true)
        # build_linearopt(500, 1000)
        build_linearopt(15, 20)

    Hypatia.check_data(c, A, b, G, h, cone)
    (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c, A, b, G, useQR=true)
    L = Hypatia.QRChol(c1, A1, b1, G1, h, cone, Q2, RiQ1)

    mdl = Hypatia.Model(maxiter=100, verbose=true)
    Hypatia.load_data!(mdl, c1, A1, b1, G1, h, cone, L)
    Hypatia.solve!(mdl)

    x = zeros(length(c))
    x[dukeep] = Hypatia.get_x(mdl)
    y = zeros(length(b))
    y[prkeep] = Hypatia.get_y(mdl)
    s = Hypatia.get_s(mdl)
    z = Hypatia.get_z(mdl)

    status = Hypatia.get_status(mdl)
    solvetime = Hypatia.get_solvetime(mdl)
    pobj = Hypatia.get_pobj(mdl)
    dobj = Hypatia.get_dobj(mdl)

    @test status == :Optimal
    # @show status
    # @show x
    # @show pobj
    # @show dobj
    return nothing
end
