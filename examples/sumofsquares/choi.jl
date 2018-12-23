#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

Example modified from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/choi.jl
Verifies that a given polynomial matrix is not a Sum-of-Squares matrix
See Choi, M. D., "Positive semidefinite biquadratic forms", Linear Algebra and its Applications, 1975, 12(2), 95-100
=#

using JuMP
using MathOptInterface
MOI = MathOptInterface
using Hypatia
using MultivariatePolynomials
using DynamicPolynomials
using PolyJuMP
using Test

function run_JuMP_choi_scalarwsos()
    @polyvar x y z
    C = [x^2+2y^2 -x*y -x*z;
         -x*y y^2+2z^2 -y*z;
         -x*z -y*z z^2+2x^2]
    @polyvar w[1:3]

    d = maximum(DynamicPolynomials.maxdegree.(C))
    dom = Hypatia.FreeDomain(3)
    full_dom = Hypatia.add_free_vars(dom)
    (U, pts, P0, _, _) = Hypatia.interpolate(full_dom, d+2, sample_factor=50, sample=true)
    scalar_wsos_cone = WSOSPolyInterpCone(U, [P0])
    conv_condition = w'*C*w

    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
    @constraint(model, [AffExpr(conv_condition(pts[i, :])) for i in 1:U] in scalar_wsos_cone)
    JuMP.optimize!(model)
    @test JuMP.dual_status(model) == MOI.INFEASIBILITY_CERTIFICATE
end
