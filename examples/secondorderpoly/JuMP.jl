#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

check a sufficient condition for pointwise membership of vector valued polynomials in the second order cone
=#

import LinearAlgebra
import Random
using Test
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import Hypatia
const HYP = Hypatia
const MU = HYP.ModelUtilities

const rt2 = sqrt(2)

function secondorderpolyJuMP(polyvec::Function, deg::Int)
    halfdeg = div(deg + 1, 2)
    (U, pts, P0, _, _) = MU.interpolate(MU.FreeDomain{Float64}(1), halfdeg, sample = false)

    vals = polyvec.(pts)
    l = length(vals[1])
    cone = HYP.WSOSPolyInterpSOCCone(l, U, [P0])

    model = JuMP.Model()
    JuMP.@constraint(model, [v[i] for i in 1:l for v in vals] in cone)

    return (model = model,)
end

secondorderpolyJuMP1() = secondorderpolyJuMP(x -> [2x^2 + 2, x, x], 2)
secondorderpolyJuMP2() = secondorderpolyJuMP(x -> [x^2 + 2, x], 2)
secondorderpolyJuMP3() = secondorderpolyJuMP(x -> [x^2 + 2, x, x], 2)
secondorderpolyJuMP4() = secondorderpolyJuMP(x -> [2 * x^4 + 8 * x^2 + 4, x + 2 + (x + 1)^2, x], 4)
secondorderpolyJuMP5() = secondorderpolyJuMP(x -> [x, x^2 + x], 2)
secondorderpolyJuMP6() = secondorderpolyJuMP(x -> [x, x + 1], 2)
secondorderpolyJuMP7() = secondorderpolyJuMP(x -> [x^2, x], 2)
secondorderpolyJuMP8() = secondorderpolyJuMP(x -> [x + 2, x], 2)
secondorderpolyJuMP9() = secondorderpolyJuMP(x -> [x - 1, x, x], 2)

function test_secondorderpolyJuMP(instance; options)
    (instance, isfeas) = instance
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(d.model) == (isfeas ? MOI.OPTIMAL : MOI.INFEASIBLE)
end

test_secondorderpolyJuMP_all(; options...) = test_secondorderpolyJuMP.([
    (secondorderpolyJuMP1, true),
    (secondorderpolyJuMP2, true),
    (secondorderpolyJuMP3, true),
    (secondorderpolyJuMP4, true),
    (secondorderpolyJuMP5, false),
    (secondorderpolyJuMP6, false),
    (secondorderpolyJuMP7, false),
    (secondorderpolyJuMP8, false),
    (secondorderpolyJuMP9, false),
    ], options = options)

test_secondorderpolyJuMP(; options...) = test_secondorderpolyJuMP.([
    (secondorderpolyJuMP1, true),
    (secondorderpolyJuMP6, false),
    ], options = options)
