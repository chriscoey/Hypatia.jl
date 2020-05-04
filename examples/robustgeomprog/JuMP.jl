#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

robust geometric programming problem
given a convex set C in R_+^k (described by conic constraints) and a matrix B in R^{k, n}, calculate
    f(C, B) = sup_{c in C} (inf_{x in R^n, z in R_+^k} c'*z : B_i*x <= log(z_i), i = 1..k)
note the inner problem is an unconstrained GP, and C specifies coefficient uncertainty
for more details, see section 4.4 of:
"Relative entropy optimization and its applications" (2017) by Chandrasekaran & Shah
the authors show that:
    f(C, B) = -inf_{c in C, v in R_+^k} (d(v, e*c) : B_j'*v = 0, j = 1..n)
where e = exp(1) and d(a, b) = sum_i a_i*log(a_i/b_i) is the relative entropy of a and b
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))

struct RobustGeomProgJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    k::Int
end

example_tests(::Type{RobustGeomProgJuMP{Float64}}, ::MinimalInstances) = [
    ((2, 3),),
    ((2, 3), ClassicConeOptimizer),
    ]
example_tests(::Type{RobustGeomProgJuMP{Float64}}, ::FastInstances) = [
    ((5, 10),),
    ((5, 10), ClassicConeOptimizer),
    ((10, 20),),
    ((20, 40),),
    ((40, 80),),
    ]
example_tests(::Type{RobustGeomProgJuMP{Float64}}, ::SlowInstances) = [
    ((40, 80), ClassicConeOptimizer),
    ((100, 200),),
    ]
example_tests(::Type{RobustGeomProgJuMP{Float64}}, ::ExpInstances) = [
    ((5, 10), ClassicConeOptimizer),
    ((10, 20), ClassicConeOptimizer),
    ((20, 40), ClassicConeOptimizer),
    ((40, 80), ClassicConeOptimizer),
    ]

function build(inst::RobustGeomProgJuMP{T}) where {T <: Float64} # TODO generic reals
    (n, k) = (inst.n, inst.k)
    @assert n < k # want some degrees of freedom for v
    B = randn(T, k, n) # linear constraint matrix
    alphas = rand(T, k) .+ 1 # for entropy constraint for set C

    model = JuMP.Model()
    JuMP.@variable(model, t)
    JuMP.@objective(model, Min, t)

    JuMP.@variable(model, c[1:k])
    JuMP.@variable(model, v[1:k])
    JuMP.@constraint(model, vcat(t, ℯ * c, v) in MOI.RelativeEntropyCone(1 + 2k))
    JuMP.@constraint(model, B' * v .== 0)

    # use bounded convex set C of R_+^k excluding origin (note that the entropy constraint already forces c >= 0)
    # satisfy a geomean constraint (note c = ones(k) is feasible and origin is excluded)
    JuMP.@constraint(model, vcat(1, c) in MOI.GeometricMeanCone(1 + k))
    # satisfy an entropy constraint with perspective vector alphas (note c = ones(k) is feasible and no c variable can go to infinity)
    @assert all(alphas .> 1e-5)
    alphas /= sum(alphas)
    JuMP.@constraint(model, vcat(-sum(log, alphas), alphas, c) in MOI.RelativeEntropyCone(1 + 2k))

    return model
end

return RobustGeomProgJuMP
