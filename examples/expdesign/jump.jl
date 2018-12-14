#=
Copyright 2018, Chris Coey and contributors

D-optimal experimental design
adapted from Boyd and Vandenberghe, "Convex Optimization", section 7.5
  maximize    logdet(V*diagm(np)*V')
  subject to  sum(np) == n
              0 .<= np .<= nmax
where np is a vector of variables representing the number of experiment p to run (fractional), and the columns of V are the vectors representing each experiment
=#

using Hypatia
using MathOptInterface
MOI = MathOptInterface
using JuMP
using Random
using Test

function build_JuMP_expdesign(
    q::Int,
    p::Int,
    V::Matrix{Float64},
    n::Int,
    nmax::Int,
    )
    @assert (p > q) && (n > q) && (nmax <= n)
    @assert size(V) == (q, p)

    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))

    @variable(model, hypo) # hypograph of logdet variable
    @objective(model, Max, hypo)

    @variable(model, 0 <= np[1:p] <= nmax) # number of each experiment
    @constraint(model, sum(np) == n) # n experiments total

    Q = V*diagm(np)*V' # information matrix
    @constraint(model, vcat(hypo, 1.0, [Q[i,j] for i in 1:q for j in 1:i]) in MOI.LogDetConeTriangle(q)) # hypograph of logdet of information matrix

    return (model, np)
end

function run_JuMP_expdesign(;rseed::Int=1)
    (q, p, n, nmax) =
        # 25, 75, 125, 5     # large
        # 10, 30, 50, 5      # medium
        5, 15, 25, 5       # small
        # 4, 8, 12, 3        # tiny
        # 3, 5, 7, 2         # miniscule

    # generate random experiment vectors
    Random.seed!(rseed)
    V = randn(q, p)

    (model, np) = build_JuMP_expdesign(q, p, V, n, nmax)
    JuMP.optimize!(model)

    term_status = JuMP.termination_status(model)
    pobj = JuMP.objective_value(model)
    dobj = JuMP.objective_bound(model)
    pr_status = JuMP.primal_status(model)
    du_status = JuMP.dual_status(model)
    npval = JuMP.value.(np)

    @test term_status == MOI.Optimal
    @test pr_status == MOI.FeasiblePoint
    @test du_status == MOI.FeasiblePoint
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4
    @test pobj ≈ logdet(V*Diagonal(npval)*V') atol=1e-4 rtol=1e-4
    @test sum(npval) ≈ n atol=1e-4 rtol=1e-4
    @test all(-1e-4 .<= npval .<= nmax + 1e-4)

    return nothing
end
