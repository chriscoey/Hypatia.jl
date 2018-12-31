#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

Given data (xᵢ, yᵢ), find a polynomial p to solve
    minimize ∑ᵢℓ(p(xᵢ), yᵢ)
    subject to ρⱼ × dᵏp/dtⱼᵏ ≥ 0 ∀ t ∈ D
where
    - dᵏp/dtⱼᵏ is the kᵗʰ derivative of p in direction j,
    - ρⱼ determines the desired sign of the derivative,
    - D is a domain such as a box or an ellipsoid,
    - ℓ is a convex loss function.
See e.g. Chapter 8 of thesis by G. Hall (2018).
=#

using Random
import Distributions
using MathOptInterface
MOI = MathOptInterface
using JuMP
using Hypatia
using MultivariatePolynomials
using DynamicPolynomials
using SumOfSquares
using PolyJuMP
using Test
include(joinpath(dirname(@__DIR__), "utils", "semialgebraicsets.jl"))


# a description of the shape of the regressor
mutable struct ShapeData
    mono_dom::Hypatia.InterpDomain
    conv_dom::Hypatia.InterpDomain
    mono_profile::Vector{Int}
    conv_profile::Int
end

ShapeData(n::Int) = ShapeData(
    Hypatia.Box(-ones(n), ones(n)),
    Hypatia.Box(-ones(n), ones(n)),
    ones(Int, n),
    1,
    )

# problem data
function generateregrdata(
    func::Function,
    xmin::Float64,
    xmax::Float64,
    n::Int,
    npoints::Int;
    signal_ratio::Float64 = 1.0,
    rseed::Int = 1,
    )
    @assert 0.0 <= signal_ratio < Inf
    Random.seed!(rseed)
    X = rand(Distributions.Uniform(xmin, xmax), npoints, n)
    y = [func(X[p,:]) for p in 1:npoints]

    if !iszero(signal_ratio)
        noise = rand(Distributions.Normal(), npoints)
        noise .*= norm(y)/sqrt(signal_ratio)/norm(noise)
        y .+= noise
    end

    return (X, y)
end

function add_loss_and_polys!(
    model::Model,
    X::Matrix{Float64},
    y::Vector{Float64},
    r::Int,
    use_leastsqobj::Bool
    )
    (npoints, n) = size(X)
    @polyvar x[1:n]
    @variable(model, p, PolyJuMP.Poly(monomials(x, 0:r)))
    if use_leastsqobj
        println("using SOC formulation")
        @variable(model, z)
        @objective(model, Min, z)
        @constraint(model, [z, [y[i] - p(X[i, :]) for i in 1:npoints]...] in MOI.SecondOrderCone(1+npoints))
     else
        println("using QP obj formulation")
        @objective(model, Min, sum((y[i] - p(X[i, :]))^2 for i in 1:npoints))
        # @variable(model, z[1:npoints])
        # @objective(model, Min, sum(z) / npoints)
        # @constraints(model, begin
        #     [i in 1:npoints], z[i] >= y[i] - p(X[i, :])
        #     [i in 1:npoints], z[i] >= -y[i] + p(X[i, :])
        # end)
    end
    return (x, p)
end

function build_shapeconregr_PSD(
    X::Matrix{Float64},
    y::Vector{Float64},
    r::Int,
    sd::ShapeData;
    use_leastsqobj::Bool = false,
    usedense::Bool = true,
    )
    n = size(X, 2)

    model = SOSModel(with_optimizer(Hypatia.Optimizer, verbose=true, usedense=usedense, lscachetype=Hypatia.QRChol))
    (x, p) = add_loss_and_polys!(model, X, y, r, use_leastsqobj)

    mono_bss = get_domain_inequalities(sd.mono_dom, x)
    conv_bss = get_domain_inequalities(sd.conv_dom, x)

    # monotonicity
    dp = [DynamicPolynomials.differentiate(p, x[i]) for i in 1:n]
    for j in 1:n
        if !iszero(sd.mono_profile[j])
            @constraint(model, sd.mono_profile[j] * dp[j] >= 0, domain=mono_bss)
        end
    end

    # convexity
    if !iszero(sd.conv_profile)
        # TODO think about what it means if wsos polynomials have degree > 2
        Hp = [DynamicPolynomials.differentiate(dp[i], x[j]) for i in 1:n, j in 1:n]
        @SDconstraint(model, sd.conv_profile * Hp >= 0, domain=conv_bss)
    end

    return (model, p)
end

function build_shapeconregr_WSOS(
    X::Matrix{Float64},
    y::Vector{Float64},
    r::Int,
    sd::ShapeData;
    use_leastsqobj::Bool = false,
    usedense::Bool = true,
    sample::Bool = true,
    )
    d = div(r, 2)
    n = size(X, 2)

    full_conv_dom = Hypatia.add_free_vars(sd.conv_dom)
    (mono_U, mono_pts, mono_P0, mono_PWts, _) = Hypatia.interpolate(sd.mono_dom, d, sample=sample, sample_factor=50)
    (conv_U, conv_pts, conv_P0, conv_PWts, _) = Hypatia.interpolate(full_conv_dom, d+1, sample=sample, sample_factor=50) # TODO think about if it's ok to go up to d+1
    mono_wsos_cone = WSOSPolyInterpCone(mono_U, [mono_P0, mono_PWts...])
    conv_wsos_cone = WSOSPolyInterpCone(conv_U, [conv_P0, conv_PWts...])

    model = SOSModel(with_optimizer(Hypatia.Optimizer, verbose=true, usedense=usedense, lscachetype=Hypatia.QRChol))
    (x, p) = add_loss_and_polys!(model, X, y, r, use_leastsqobj)
    @polyvar w[1:n]

    # monotonicity
    dp = [DynamicPolynomials.differentiate(p, x[j]) for j in 1:n]
    for j in 1:n
        if !iszero(sd.mono_profile[j])
            @constraint(model, [sd.mono_profile[j] * dp[j](mono_pts[i, :]) for i in 1:mono_U] in mono_wsos_cone)
        end
    end

    # convexity
    if !iszero(sd.conv_profile)
        Hp = [DynamicPolynomials.differentiate(dp[i], x[j]) for i in 1:n, j in 1:n]
        conv_condition = w'*Hp*w
        @constraint(model, [sd.conv_profile * conv_condition(conv_pts[i, :]) for i in 1:conv_U] in conv_wsos_cone)
    end

    return (model, p)
end

function run_JuMP_shapeconregr(
    use_wsos::Bool;
    use_leastsqobj::Bool = true,
    usedense::Bool = true,
    )
    (n, deg, npoints, signal_ratio, f) =
        # (2, 3, 100, 0.0, x -> exp(norm(x))) # no noise, monotonic function
        # (2, 3, 100, 0.0, x -> sum(x.^3)) # no noise, monotonic function
        (2, 3, 100, 0.0, x -> sum(x.^3)) # no noise, monotonic function
        # (2, 3, 100, 0.0, x -> sum(x.^4)) # no noise, non-monotonic function
        # (2, 3, 100, 50.0, x -> sum(x.^3)) # some noise, monotonic function
        # (2, 3, 100, 50.0, x -> sum(x.^4)) # some noise, non-monotonic function
        # (2, 6, 100, 0.0, x -> exp(norm(x))) # low n high deg, numerically harder
        # (5, 5, 100, 0.0, x -> exp(norm(x))) # moderate size, no noise, monotonic # out of memory with psd

    shapedata = ShapeData(n)
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)

    if use_wsos
        (model, p) = build_shapeconregr_WSOS(X, y, deg, shapedata, use_leastsqobj=use_leastsqobj, usedense=usedense)
    else
        (model, p) = build_shapeconregr_PSD(X, y, deg, shapedata, use_leastsqobj=use_leastsqobj, usedense=usedense)
    end

    println("starting to solve JuMP model")
    JuMP.optimize!(model)
    term_status = JuMP.termination_status(model)
    pobj = JuMP.objective_value(model)
    dobj = JuMP.objective_bound(model)
    pr_status = JuMP.primal_status(model)
    du_status = JuMP.dual_status(model)

    @test term_status == MOI.OPTIMAL
    @test pr_status == MOI.FEASIBLE_POINT
    @test du_status == MOI.FEASIBLE_POINT
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4

    return (pobj, p)
end

run_JuMP_shapeconregr_PSD_abs() = run_JuMP_shapeconregr(false, use_leastsqobj=false)
run_JuMP_shapeconregr_WSOS_abs() = run_JuMP_shapeconregr(true, use_leastsqobj=false)
run_JuMP_shapeconregr_PSD_lsq() = run_JuMP_shapeconregr(false, use_leastsqobj=true)
run_JuMP_shapeconregr_WSOS_lsq() = run_JuMP_shapeconregr(true, use_leastsqobj=true)
