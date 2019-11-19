#=
Copyright 2018, Chris Coey and contributors

see description in examples/envelope/native.jl
=#

using LinearAlgebra
import Random
using Test
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import Hypatia
const HYP = Hypatia
const MU = HYP.ModelUtilities

function envelopeJuMP(
    npoly::Int,
    rand_halfdeg::Int,
    env_halfdeg::Int,
    domain::MU.Domain;
    sample::Bool = true,
    )
    # generate interpolation
    @assert rand_halfdeg <= env_halfdeg
    (U, pts, P0, PWts, w) = MU.interpolate(domain, env_halfdeg, sample = sample, calc_w = true)

    # generate random polynomials
    n = MU.get_dimension(domain)
    L = binomial(n + rand_halfdeg, n)
    polys = P0[:, 1:L] * rand(-9:9, L, npoly)

    model = JuMP.Model()
    JuMP.@variable(model, fpv[j in 1:U]) # values at Fekete points
    JuMP.@objective(model, Max, dot(fpv, w)) # integral over domain (via quadrature)
    JuMP.@constraint(model, [i in 1:npoly], polys[:, i] .- fpv in HYP.WSOSPolyInterpCone(U, [P0, PWts...]))

    return (model = model,)
end

envelopeJuMP1() = envelopeJuMP(2, 3, 4, MU.Box{Float64}(-ones(2), ones(2)))
# envelopeJuMP2() = envelopeJuMP(2, 3, 4, MU.Ball{Float64}(zeros(2), sqrt(2))) # TODO needs https://github.com/chriscoey/Hypatia.jl/issues/173
envelopeJuMP3() = envelopeJuMP(2, 3, 4, MU.Box{Float64}(-ones(2), ones(2)), sample = false)

function test_envelopeJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_envelopeJuMP_all(; options...) = test_envelopeJuMP.([
    envelopeJuMP1,
    # envelopeJuMP2,
    envelopeJuMP3,
    ], options = options)

test_envelopeJuMP(; options...) = test_envelopeJuMP.([
    envelopeJuMP1,
    # envelopeJuMP2,
    envelopeJuMP3,
    ], options = options)
