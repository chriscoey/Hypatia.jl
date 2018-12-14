#=
Copyright 2018, Chris Coey and contributors

see description in examples/envelope/native.jl
=#

using Hypatia
import MathOptInterface
MOI = MathOptInterface
using JuMP
using LinearAlgebra
using Random
using Test

function build_JuMP_envelope(
    npoly::Int,
    deg::Int,
    d::Int,
    domain::Hypatia.InterpDomain;
    sample::Bool = true,
    rseed::Int = 1,
    )
    # generate interpolation
    @assert deg <= d
    (U, pts, P0, PWts, w) = Hypatia.interpolate(domain, d, sample=sample, calc_w=true)

    # generate random polynomials
    Random.seed!(rseed)
    n = Hypatia.dimension(domain)
    LDegs = binomial(n+deg, n)
    polys = P0[:, 1:LDegs]*rand(-9:9, LDegs, npoly)

    # build JuMP model
    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
    @variable(model, fpv[j in 1:U]) # values at Fekete points
    @objective(model, Max, dot(fpv, w)) # integral over domain (via quadrature)
    @constraint(model, [i in 1:npoly], polys[:,i] .- fpv in WSOSPolyInterpCone(U, [P0, PWts...]))

    return (model, fpv)
end

function run_JuMP_envelope(
    npoly::Int,
    deg::Int,
    d::Int,
    dom::Hypatia.InterpDomain;
    sample::Bool = true,
    )
    (model, fpv) = build_JuMP_envelope(npoly, deg, d, dom, sample=sample)
    JuMP.optimize!(model)

    term_status = JuMP.termination_status(model)
    pobj = JuMP.objective_value(model)
    dobj = JuMP.objective_bound(model)
    pr_status = JuMP.primal_status(model)
    du_status = JuMP.dual_status(model)

    @test term_status == MOI.Optimal
    @test pr_status == MOI.FeasiblePoint
    @test du_status == MOI.FeasiblePoint
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4

    return nothing
end

run_JuMP_envelope_sampleinterp_box() = run_JuMP_envelope(2, 3, 4, Hypatia.Box(-ones(2), ones(2)))
run_JuMP_envelope_sampleinterp_ball() = run_JuMP_envelope(2, 3, 4, Hypatia.Ball(zeros(2), sqrt(2)))
run_JuMP_envelope_boxinterp() = run_JuMP_envelope(2, 3, 4, Hypatia.Box(-ones(2), ones(2)), sample=false)
