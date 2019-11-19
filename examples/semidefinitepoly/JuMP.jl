#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

test whether a given matrix has a SOS decomposition,
and use this procedure to check whether a polynomial is globally convex
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

function semidefinitepolyJuMP(x::Vector{DP.PolyVar{true}}, H::Matrix; use_wsos::Bool = true, use_dual::Bool = false)
    model = JuMP.Model()

    if use_wsos
        matdim = size(H, 1)
        halfdeg = div(maximum(DP.maxdegree.(H)) + 1, 2)
        n = DP.nvariables(x)
        dom = MU.FreeDomain{Float64}(n)
        (U, pts, P0, _, _) = MU.interpolate(dom, halfdeg, sample_factor = 20, sample = true)
        mat_wsos_cone = HYP.WSOSPolyInterpMatCone(matdim, U, [P0], use_dual)

        if use_dual
            JuMP.@variable(model, z[i in 1:n, 1:i, 1:U])
            JuMP.@constraint(model, [z[i, j, u] * (i == j ? 1.0 : rt2) for i in 1:matdim for j in 1:i for u in 1:U] in mat_wsos_cone)
            JuMP.@objective(model, Min, sum(z[i, j, u] * H[i, j](pts[u, :]...) * (i == j ? 1.0 : 2.0) for i in 1:matdim for j in 1:i for u in 1:U))
        else
            JuMP.@constraint(model, [H[i, j](pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:matdim for j in 1:i for u in 1:U] in mat_wsos_cone)
        end
    else
        if use_dual
            error("dual formulation not implemented for scalar SOS formulation")
        else
            PolyJuMP.setpolymodule!(model, SumOfSquares)
            JuMP.@constraint(model, H in JuMP.PSDCone())
        end
    end

    return (model = model,)
end

semidefinitepolyJuMP(x::Vector{DP.PolyVar{true}}, poly::DP.Polynomial; use_wsos::Bool = true, use_dual::Bool = false) =
    semidefinitepolyJuMP(x, DP.differentiate(poly, x, 2), use_wsos = use_wsos, use_dual = use_dual)

function semidefinitepolyJuMP1()
    DP.@polyvar x
    M = [
        (x + 2x^3)  1;
        (-x^2 + 2)  (3x^2 - x + 1);
        ]
    MM = M' * M
    return semidefinitepolyJuMP([x], MM, use_wsos = true)
end

function semidefinitepolyJuMP2()
    DP.@polyvar x
    poly = x^4 + 2x^2
    return semidefinitepolyJuMP([x], poly, use_wsos = true)
end

function semidefinitepolyJuMP3()
    DP.@polyvar x y
    poly = (x + y)^4 + (x + y)^2
    return semidefinitepolyJuMP([x, y], poly, use_wsos = true)
end

function semidefinitepolyJuMP4()
    n = 3
    m = 3
    d = 1
    DP.@polyvar x[1:n]
    Z = DP.monomials(x, 0:d)
    M = [sum(rand() * Z[l] for l in 1:length(Z)) for i in 1:m, j in 1:m]
    MM = M' * M
    MM = 0.5 * (MM + MM')
    return semidefinitepolyJuMP(x, MM, use_wsos = true)
end

# example modified from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/simplematrixsos.jl
# Example 3.77 and 3.79 of Blekherman, G., Parrilo, P. A., & Thomas, R. R. (Eds.),
# Semidefinite optimization and convex algebraic geometry SIAM 2013
function semidefinitepolyJuMP5()
    DP.@polyvar x
    P = [
        (x^2 - 2x + 2)  x;
        x               x^2;
        ]
    return semidefinitepolyJuMP([x], P, use_wsos = true)
end

# example modified from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/choi.jl
# verifies that a given polynomial matrix is not a Sum-of-Squares matrix
# see Choi, M. D., "Positive semidefinite biquadratic forms",
# Linear Algebra and its Applications, 1975, 12(2), 95-100
function semidefinitepolyJuMP6()
    DP.@polyvar x y z
    P = [
        (x^2 + 2y^2)    (-x * y)        (-x * z);
        (-x * y)        (y^2 + 2z^2)    (-y * z);
        (-x * z)        (-y * z)        (z^2 + 2x^2);
        ] .* (x * y * z)^0 # TODO the (x * y * z)^0 can be removed when https://github.com/JuliaOpt/SumOfSquares.jl/issues/106 is fixed
    return semidefinitepolyJuMP([x, y, z], P, use_wsos = true)
end

# example modified from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/sosdemo9.jl
# Section 3.9 of SOSTOOLS User's Manual, see https://www.cds.caltech.edu/sostools/
function semidefinitepolyJuMP7()
    DP.@polyvar x y z
    P = [
        (x^4 + x^2 * y^2 + x^2 * z^2)                       (x * y * z^2 - x^3 * y - x * y * (y^2 + 2 * z^2));
        (x * y * z^2 - x^3 * y - x * y * (y^2 + 2 * z^2))   (x^2 * y^2 + y^2 * z^2 + (y^2 + 2 * z^2)^2);
        ]
    return semidefinitepolyJuMP([x, y, z], P, use_wsos = true)
end

function test_semidefinitepolyJuMP(instance::Tuple{Function,Bool}; options, rseed::Int = 1)
    Random.seed!(1)
    (instance, is_feas) = instance
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(d.model) == (is_feas ? MOI.OPTIMAL : MOI.INFEASIBLE)
end

test_semidefinitepolyJuMP_all(; options...) = test_semidefinitepolyJuMP.([
    (semidefinitepolyJuMP1, true),
    (semidefinitepolyJuMP2, true),
    (semidefinitepolyJuMP3, true),
    (semidefinitepolyJuMP4, true),
    (semidefinitepolyJuMP5, true),
    (semidefinitepolyJuMP6, false),
    (semidefinitepolyJuMP7, true),
    ], options = options)

test_semidefinitepolyJuMP(; options...) = test_semidefinitepolyJuMP.([
    (semidefinitepolyJuMP1, true),
    (semidefinitepolyJuMP3, true),
    (semidefinitepolyJuMP5, true),
    (semidefinitepolyJuMP6, false),
    (semidefinitepolyJuMP7, true),
    ], options = options)
