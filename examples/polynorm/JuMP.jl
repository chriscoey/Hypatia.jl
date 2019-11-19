#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find a polynomial f such that f² >= Σᵢ gᵢ² where gᵢ are arbitrary polynomials, and the volume under f is minimized

TODO add scalar SOS formulation
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

function polynormJuMP(n::Int, deg::Int, npolys::Int)
    dom = MU.FreeDomain{Float64}(n)
    halfdeg = div(deg + 1, 2)
    (U, pts, P0, _, w) = MU.interpolate(dom, halfdeg, sample = false, calc_w = true)
    lagrange_polys = MU.recover_lagrange_polys(pts, 2 * halfdeg)

    model = JuMP.Model()
    JuMP.@variable(model, f[1:U])
    JuMP.@objective(model, Min, dot(w, f))

    L = size(P0, 2)
    polys = P0[:, 1:L] * rand(-9:9, L, npolys)

    fpoly = dot(f, lagrange_polys)
    rand_polys = [dot(polys[:, i], lagrange_polys) for i in 1:npolys]
    cone = HYP.WSOSPolyInterpSOCCone(npolys + 1, U, [P0])
    JuMP.@constraint(model, vcat(f, [polys[:, i] for i in 1:npolys]...) in cone)

    return (model = model,)
end

polynormJuMP1() = polynormJuMP(2, 2, 2)
polynormJuMP2() = polynormJuMP(2, 1, 3)

function test_polynormJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_polynormJuMP_all(; options...) = test_polynormJuMP.([
    polynormJuMP1,
    polynormJuMP2,
    ], options = options)

test_polynormJuMP(; options...) = test_polynormJuMP.([
    polynormJuMP1,
    ], options = options)
