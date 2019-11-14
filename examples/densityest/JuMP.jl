#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

see description in native.jl
==#

using LinearAlgebra
import Random
using Test
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import DynamicPolynomials
import PolyJuMP
import Hypatia
const HYP = Hypatia
const MU = HYP.ModelUtilities

include(joinpath(@__DIR__, "data.jl"))

function densityestJuMP(
    X::Matrix{Float64},
    deg::Int;
    sample_factor::Int = 100,
    use_monomials::Bool = false,
    )
    (nobs, dim) = size(X)

    domain = MU.Box{Float64}(-ones(dim), ones(dim))
    # rescale X to be in unit box
    minX = minimum(X, dims = 1)
    maxX = maximum(X, dims = 1)
    X .-= (minX + maxX) / 2
    X ./= (maxX - minX) / 2

    halfdeg = div(deg + 1, 2)
    (U, pts, P0, PWts, w) = MU.interpolate(domain, halfdeg, sample = true, calc_w = true, sample_factor = sample_factor)
    @show typeof(P0), typeof(PWts), typeof(w)

    model = JuMP.Model()
    JuMP.@variable(model, z[1:nobs])
    JuMP.@objective(model, Max, sum(z))

    if use_monomials
        lagrange_polys = []
        DynamicPolynomials.@polyvar x[1:dim]
        PX = DynamicPolynomials.monomials(x, 0:(2 * halfdeg))

        JuMP.@variable(model, f, PolyJuMP.Poly(PX))
        JuMP.@constraints(model, begin
            sum(w[i] * f(pts[i, :]) for i in 1:U) == 1.0 # integrate to 1
            [f(pts[i, :]) for i in 1:U] in HYP.WSOSPolyInterpCone(U, [P0, PWts...]) # density nonnegative
            [i in 1:nobs], vcat(z[i], 1.0, f(X[i, :])) in MOI.ExponentialCone() # hypograph of log
        end)
    else
        lagrange_polys = MU.recover_lagrange_polys(pts, 2 * halfdeg)
        basis_evals = Matrix{Float64}(undef, nobs, U)
        for i in 1:nobs, j in 1:U
            basis_evals[i, j] = lagrange_polys[j](X[i, :])
        end

        JuMP.@variable(model, f[1:U])
        JuMP.@constraints(model, begin
            dot(w, f) == 1.0 # integrate to 1
            f in HYP.WSOSPolyInterpCone(U, [P0, PWts...]) # density nonnegative
            [i in 1:nobs], vcat(z[i], 1.0, dot(f, basis_evals[i, :])) in MOI.ExponentialCone() # hypograph of log
        end)
    end

    return (model = model,)
end

densityestJuMP(nobs::Int, n::Int, deg::Int, use_monomials::Bool) = densityestJuMP(randn(nobs, n), deg, use_monomials = use_monomials)

densityestJuMP1() = densityestJuMP(iris_data(), 4)
densityestJuMP2() = densityestJuMP(iris_data(), 6)
densityestJuMP3() = densityestJuMP(cancer_data(), 4)
densityestJuMP4() = densityestJuMP(cancer_data(), 6)
densityestJuMP5() = densityestJuMP(200, 1, 4, false)
densityestJuMP6() = densityestJuMP(200, 1, 4, true)

function test_densityestJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    @show "have model"
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_densityestJuMP_all(; options...) = test_densityestJuMP.([
    densityestJuMP1,
    densityestJuMP2,
    densityestJuMP3,
    densityestJuMP4,
    densityestJuMP5,
    densityestJuMP6,
    ], options = options)

test_densityestJuMP(; options...) = test_densityestJuMP.([
    densityestJuMP1,
    densityestJuMP3,
    densityestJuMP5,
    densityestJuMP6,
    ], options = options)
