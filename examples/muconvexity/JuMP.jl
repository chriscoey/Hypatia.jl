#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find parameter of convexity mu for a given polynomial p(x)
ie the largest mu such that p(x) - mu/2*||x||^2 is convex everywhere on given domain
see https://en.wikipedia.org/wiki/Convex_function#Strongly_convex_functions
=#

using Test
import Random
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import DynamicPolynomials
const DP = DynamicPolynomials
import SumOfSquares
import PolyJuMP
import Hypatia
const HYP = Hypatia
const MU = HYP.ModelUtilities

const rt2 = sqrt(2)

function muconvexityJuMP(
    polyfun::Function,
    dom::MU.Domain;
    use_wsos::Bool = true,
    )
    n = MU.get_dimension(dom)
    DP.@polyvar x[1:n]
    poly = polyfun(x)

    model = JuMP.Model()
    JuMP.@variable(model, mu)
    JuMP.@objective(model, Max, mu)

    convpoly = poly - 0.5 * mu * sum(x.^2)
    H = DP.differentiate(convpoly, x, 2)

    if use_wsos
        d = div(maximum(DP.maxdegree.(H)) + 1, 2)
        (U, pts, P0, PWts, _) = MU.interpolate(dom, d, sample = true, sample_factor = 100)
        mat_wsos_cone = HYP.WSOSPolyInterpMatCone(n, U, [P0, PWts...])
        JuMP.@constraint(model, [H[i, j](x => pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:n for j in 1:i for u in 1:U] in mat_wsos_cone)
    else
        PolyJuMP.setpolymodule!(model, SumOfSquares)
        JuMP.@constraint(model, H in JuMP.PSDCone(), domain = MU.get_domain_inequalities(dom, x))
    end

    return (model = model, mu = mu)
end

muconvexityJuMP1() = muconvexityJuMP(x -> (x[1] + 1)^2 * (x[1] - 1)^2, MU.FreeDomain{Float64}(1), use_wsos = true)
muconvexityJuMP2() = muconvexityJuMP(x -> sum(x.^4) - sum(x.^2), MU.FreeDomain{Float64}(3), use_wsos = true)
muconvexityJuMP3() = muconvexityJuMP(x -> (x[1] + 1)^2 * (x[1] - 1)^2, MU.Box{Float64}([-1.0], [1.0]), use_wsos = true)
muconvexityJuMP4() = muconvexityJuMP(x -> sum(x.^4) - sum(x.^2), MU.Ball{Float64}(ones(2), 5.0), use_wsos = true)
muconvexityJuMP5() = muconvexityJuMP(x -> (x[1] + 1)^2 * (x[1] - 1)^2, MU.FreeDomain{Float64}(1), use_wsos = false)
muconvexityJuMP6() = muconvexityJuMP(x -> sum(x.^4) - sum(x.^2), MU.FreeDomain{Float64}(3), use_wsos = false)
muconvexityJuMP7() = muconvexityJuMP(x -> (x[1] + 1)^2 * (x[1] - 1)^2, MU.Box{Float64}([-1.0], [1.0]), use_wsos = false)
muconvexityJuMP8() = muconvexityJuMP(x -> sum(x.^4) - sum(x.^2), MU.Ball{Float64}(ones(2), 5.0), use_wsos = false)

function test_muconvexityJuMP(instance::Tuple{Function, Float64}; options, rseed::Int = 1)
    Random.seed!(rseed)
    (instance, true_mu) = instance
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.value(d.mu) ≈ true_mu atol = 1e-4 rtol = 1e-4
end

test_muconvexityJuMP_all(; options...) = test_muconvexityJuMP.([
    (muconvexityJuMP1, -4.0),
    (muconvexityJuMP2, -2.0),
    (muconvexityJuMP3, -4.0),
    (muconvexityJuMP4, -2.0),
    (muconvexityJuMP5, -4.0),
    (muconvexityJuMP6, -2.0),
    (muconvexityJuMP7, -4.0),
    (muconvexityJuMP8, -2.0),
    ], options = options)

test_muconvexityJuMP(; options...) = test_muconvexityJuMP.([
    (muconvexityJuMP1, -4.0),
    (muconvexityJuMP3, -4.0),
    (muconvexityJuMP4, -2.0),
    (muconvexityJuMP5, -4.0),
    (muconvexityJuMP7, -4.0),
    (muconvexityJuMP8, -2.0),
    ], options = options)
