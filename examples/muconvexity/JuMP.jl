#=
find parameter of convexity mu for a given polynomial p(x)
ie the largest mu such that p(x) - mu/2*||x||^2 is convex on given domain
see https://en.wikipedia.org/wiki/Convex_function#Strongly_convex_functions
=#

import DynamicPolynomials
const DP = DynamicPolynomials
import SemialgebraicSets
const SAS = SemialgebraicSets
import SumOfSquares
import PolyJuMP

struct MuConvexityJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    poly::Symbol
    dom::Symbol
    use_matrixwsos::Bool # use wsosinterpposemideftricone, else PSD formulation
    true_mu::Real # optional true value of parameter for testing only
end

function build(inst::MuConvexityJuMP{T}) where {T <: Float64}
    dom = muconvexity_data[inst.dom]
    n = PolyUtils.dimension(dom)
    DP.@polyvar x[1:n]
    poly = muconvexity_data[inst.poly](x)

    model = JuMP.Model()
    JuMP.@variable(model, mu)
    JuMP.@objective(model, Max, mu)

    convpoly = poly - 0.5 * mu * sum(x.^2)
    H = DP.differentiate(convpoly, x, 2)

    if inst.use_matrixwsos
        d = div(maximum(DP.maxdegree.(H)) + 1, 2)
        (U, pts, Ps) = PolyUtils.interpolate(dom, d)
        mat_wsos_cone = Hypatia.WSOSInterpPosSemidefTriCone{T}(n, U, Ps)
        H_interp = [H[i, j](x => pts[u, :]) for i in 1:n for j in 1:i for u in 1:U]
        JuMP.@constraint(model, Cones.vec_to_svec!(H_interp,
            rt2 = sqrt(2), incr = U) in mat_wsos_cone)
    else
        PolyJuMP.setpolymodule!(model, SumOfSquares)
        JuMP.@constraint(model, H in JuMP.PSDCone(), domain =
            get_domain_inequalities(dom, x))
    end

    return model
end

function test_extra(inst::MuConvexityJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    if JuMP.termination_status(model) == MOI.OPTIMAL && !isnan(inst.true_mu)
        # check objective value is correct
        tol = eps(T)^0.25
        @test JuMP.objective_value(model) ≈ inst.true_mu atol = tol rtol = tol
    end
end

# construct domain inequalities for SumOfSquares models from Hypatia domains
bss() = SAS.BasicSemialgebraicSet{Float64,
    DynamicPolynomials.Polynomial{true, Float64}}()

function get_domain_inequalities(dom::PolyUtils.BoxDomain, x)
    box = bss()
    for (xi, ui, li) in zip(x, dom.u, dom.l)
        SAS.addinequality!(box, (-xi + ui) * (xi - li))
    end
    return box
end

get_domain_inequalities(dom::PolyUtils.FreeDomain, x) = bss()

muconvexity_data = Dict(
    :poly1 => (x -> (x[1] + 1)^2 * (x[1] - 1)^2),
    :poly2 => (x -> sum(x .^ 4) - sum(x .^ 2)),
    :dom1 => PolyUtils.FreeDomain{Float64}(1),
    :dom2 => PolyUtils.BoxDomain{Float64}([-1.0], [1.0]),
    :dom3 => PolyUtils.FreeDomain{Float64}(3),
    :dom4 => PolyUtils.BoxDomain{Float64}([-1.0, 0.0], [1.0, 2.0]),
    )
