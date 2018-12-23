#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

Given a sequence of observations X₁,...,Xₙ with each Xᵢ in Rᵈ, find a density function f maximizing the log likelihood of the observations.
    minimize -∑ᵢzᵢ
    subject to -zᵢ + log(f(Xᵢ)) ≥ 0 ∀ i = 1,...,n
    ∫f = 1
    f ≥ 0
where
    - zᵢ is in the hypograph of the log function of f
==#

using LinearAlgebra
import Random
import Distributions
using JuMP
using MathOptInterface
MOI = MathOptInterface
using PolyJuMP
using MultivariatePolynomials
using DynamicPolynomials
using Hypatia
using Test

function build_JuMP_densityest(
    X::Matrix{Float64},
    deg::Int,
    dom::Hypatia.InterpDomain;
    sample_factor::Int = 100,
    )
    (nobs, dim) = size(X)
    d = div(deg, 2)

    (U, pts, P0, PWts, w) = Hypatia.interpolate(dom, d, sample=true, calc_w=true, sample_factor=sample_factor)

    @polyvar x[1:dim]
    PX = PolyJuMP.monomials(x, 1:deg)
    U = size(pts, 1)

    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))

    @variables(model, begin
        z[1:nobs] # log(f(u_i)) at each observation
        f, PolyJuMP.Poly(PX) # probability density function
    end)

    @constraints(model, begin
        sum(w[i] * f(pts[i,:]) for i in 1:U) == 1.0 # integrate to 1
        [f(pts[i,:]) for i in 1:U] in Hypatia.WSOSPolyInterpCone(U, [P0, PWts...]) # density nonnegative
        [i in 1:nobs], vcat(z[i], 1.0, f(X[i,:])) in MOI.ExponentialCone() # hypograph of log
    end)

    @objective(model, Max, sum(z)) # maximize log likelihood

    return model
end

function run_JuMP_densityest(; rseed::Int=1)
    nobs = 200
    n = 1
    deg = 4

    Random.seed!(rseed)
    X = rand(Distributions.Uniform(-1, 1), nobs, n)
    dom = Hypatia.Box(-ones(n), ones(n))

    model = build_JuMP_densityest(X, deg, dom)
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

    return nothing
end
