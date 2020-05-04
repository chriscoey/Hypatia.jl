#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

compute a gram matrix of a polynomial, minimizing its log-determinant or root-determinant (equivalent optimal solutions with different optimal objective values)
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))
import DynamicPolynomials

struct CentralPolyMatJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int # number of polyvars
    halfdeg::Int # half degree of random polynomials
    logdet_obj::Bool # use logdet, else rootdet
end

example_tests(::Type{CentralPolyMatJuMP{Float64}}, ::MinimalInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    return [
    ((1, 2, false), nothing, options),
    ((1, 2, false), ClassicConeOptimizer, options),
    ((2, 2, true), nothing, options),
    ((2, 2, true), ClassicConeOptimizer, options),
    ]
end
example_tests(::Type{CentralPolyMatJuMP{Float64}}, ::FastInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    return [
    ((1, 10, true), nothing, options),
    ((1, 10, true), ClassicConeOptimizer, options),
    ((1, 15, false), nothing, options),
    ((1, 15, false), ClassicConeOptimizer, options),
    ((2, 3, true), nothing, options),
    ((2, 3, true), ClassicConeOptimizer, options),
    ((2, 3, false), nothing, options),
    ((2, 3, false), ClassicConeOptimizer, options),
    ((2, 6, true), nothing, options),
    ((2, 5, true), ClassicConeOptimizer, options),
    ((2, 7, false), nothing, options),
    ((2, 6, false), ClassicConeOptimizer, options),
    ((3, 2, true), nothing, options),
    ((3, 2, false), nothing, options),
    ((3, 4, true), nothing, options),
    ((3, 4, false), nothing, options),
    ((7, 2, true), nothing, options),
    ((7, 2, true), ClassicConeOptimizer, options),
    ((7, 2, false), nothing, options),
    ((7, 2, false), ClassicConeOptimizer, options),
    ]
end
example_tests(::Type{CentralPolyMatJuMP{Float64}}, ::SlowInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    return [
    ((1, 20, false), nothing, options),
    ((2, 3, false), nothing, options),
    ((2, 10, false), nothing, options),
    ((2, 8, false), ClassicConeOptimizer, options),
    ((3, 4, true), ClassicConeOptimizer, options),
    ((3, 4, false), ClassicConeOptimizer, options),
    ((3, 5, true), nothing, options),
    ((3, 5, false), nothing, options),
    ((6, 3, true), nothing, options),
    ((6, 3, false), nothing, options),
    ]
end
example_tests(::Type{CentralPolyMatJuMP{Float64}}, ::ExpInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    return [
    ((1, 10, true), ClassicConeOptimizer, options),
    ((2, 3, true), ClassicConeOptimizer, options),
    ((2, 5, true), ClassicConeOptimizer, options),
    ((7, 2, true), ClassicConeOptimizer, options),
    ]
end

function build(inst::CentralPolyMatJuMP{T}) where {T <: Float64} # TODO generic reals
    (n, halfdeg) = (inst.n, inst.halfdeg)

    DynamicPolynomials.@polyvar x[1:n]
    basis = DynamicPolynomials.monomials(x, 0:halfdeg)
    L = length(basis)
    poly_half = randn(L, L) / L * basis
    poly_rand = poly_half' * poly_half

    model = JuMP.Model()
    JuMP.@variable(model, hypo)
    JuMP.@objective(model, Max, hypo)

    # objective hypograph
    JuMP.@variable(model, Q[1:L, 1:L], Symmetric)
    v1 = [Q[i, j] for i in 1:L for j in 1:i] # vectorized Q
    if inst.logdet_obj
        JuMP.@constraint(model, vcat(hypo, 1, v1) in MOI.LogDetConeTriangle(L))
    else
        JuMP.@constraint(model, vcat(hypo, v1) in MOI.RootDetConeTriangle(L))
    end

    # coefficients equal
    poly_Q = basis' * Q * basis
    JuMP.@constraint(model, DynamicPolynomials.coefficients(poly_Q - poly_rand) .== 0)

    return model
end

return CentralPolyMatJuMP
