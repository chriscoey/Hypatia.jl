#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

contraction analysis example adapted from
"Stability and robustness analysis of nonlinear systems via contraction metrics and SOS programming"
Aylward, E.M., Parrilo, P.A. and Slotine, J.J.E
=#

using LinearAlgebra
using Test
import Random
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import DynamicPolynomials
const DP = DynamicPolynomials
import PolyJuMP
const PJ = PolyJuMP
import SumOfSquares
import Hypatia
const HYP = Hypatia
const MU = HYP.ModelUtilities

const rt2 = sqrt(2)

function contractionJuMP(
    beta::Float64,
    M_deg::Int,
    delta::Float64;
    use_wsos::Bool = true,
    )
    n = 2
    dom = MU.FreeDomain{Float64}(n)

    M_halfdeg = div(M_deg + 1, 2)
    (U_M, pts_M, P0_M, _, _) = MU.interpolate(dom, M_halfdeg, sample = false)
    lagrange_polys = MU.recover_lagrange_polys(pts_M, 2 * M_halfdeg)

    polyjump_basis = PJ.FixedPolynomialBasis(lagrange_polys)
    x = DP.variables(lagrange_polys[1])

    # dynamics according to the Moore-Greitzer model
    dx1dt = -x[2] - 1.5 * x[1]^2 - 0.5 * x[1]^3
    dx2dt = 3 * x[1] - x[2]
    dynamics = [dx1dt; dx2dt]

    model = JuMP.Model()
    JuMP.@variable(model, polys[1:3], PJ.Poly(polyjump_basis))

    M = [polys[1] polys[2]; polys[2] polys[3]]
    dMdt = [JuMP.dot(DP.differentiate(M[i, j], x), dynamics) for i in 1:n, j in 1:n]
    dfdx = DP.differentiate(dynamics, x)'
    Mdfdx = [sum(M[i, k] * dfdx[k, j] for k in 1:n) for i in 1:n, j in 1:n]
    R = Mdfdx + Mdfdx' + dMdt + beta * M

    if use_wsos
        deg_R = maximum(DP.maxdegree.(R))
        d_R = div(deg_R + 1, 2)
        (U_R, pts_R, P0_R, _, _) = MU.interpolate(dom, d_R, sample = true)
        JuMP.@constraint(model, [M[i, j](pts_M[u, :]) * (i == j ? 1.0 : rt2) - (i == j ? delta : 0.0) for i in 1:n for j in 1:i for u in 1:U_M] in HYP.WSOSPolyInterpMatCone(n, U_M, [P0_M]))
        JuMP.@constraint(model, [-R[i, j](pts_R[u, :]) * (i == j ? 1.0 : rt2) - (i == j ? delta : 0.0) for i in 1:n for j in 1:i for u in 1:U_R] in HYP.WSOSPolyInterpMatCone(n, U_R, [P0_R]))
    else
        PJ.setpolymodule!(model, SumOfSquares)
        JuMP.@constraint(model, M - Matrix(delta * I, n, n) in JuMP.PSDCone())
        JuMP.@constraint(model, -R - Matrix(delta * I, n, n) in JuMP.PSDCone())
    end

    return (model = model,)
end

contractionJuMP1() = contractionJuMP(0.77, 4, 1e-3, use_wsos = true)
contractionJuMP2() = contractionJuMP(0.77, 4, 1e-3, use_wsos = false)
contractionJuMP3() = contractionJuMP(0.85, 4, 1e-3, use_wsos = true)
contractionJuMP4() = contractionJuMP(0.85, 4, 1e-3, use_wsos = false)

function test_contractionJuMP(instance::Tuple{Function, Bool}; options, rseed::Int = 1)
    Random.seed!(rseed)
    (instance, is_feas) = instance
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(d.model) == (is_feas ? MOI.OPTIMAL : MOI.INFEASIBLE)
    return
end

test_contractionJuMP_all(; options...) = test_contractionJuMP.([
    (contractionJuMP1, true),
    (contractionJuMP2, true),
    (contractionJuMP3, false),
    (contractionJuMP4, false),
    ], options = options)

test_contractionJuMP(; options...) = test_contractionJuMP.([
    (contractionJuMP1, true),
    (contractionJuMP2, true),
    (contractionJuMP3, false),
    (contractionJuMP4, false),
    ], options = options)
