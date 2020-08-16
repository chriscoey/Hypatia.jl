#=
Copyright 2018, Chris Coey and contributors

D-optimal experimental design
adapted from Boyd and Vandenberghe, "Convex Optimization", section 7.5
  maximize    F(V*diagm(np)*V')
  subject to  sum(np) == n
              0 .<= np .<= n_max
where np is a vector of variables representing the number of experiment p to run (fractional),
and the columns of V are the vectors representing each experiment

if logdet_obj or rootdet_obj is true, F is the logdet or rootdet function
if geomean_obj is true, we use a formulation from https://picos-api.gitlab.io/picos/optdes.html that finds an equivalent minimizer
=#

struct ExpDesignJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    q::Int
    p::Int
    n::Int
    n_max::Int
    logdet_obj::Bool # use formulation with logdet objective
    rootdet_obj::Bool # use formulation with rootdet objective
    geomean_obj::Bool # use formulation with geomean objective
end

function build(inst::ExpDesignJuMP{T}) where {T <: Float64} # TODO generic reals
    (q, p, n, n_max) = (inst.q, inst.p, inst.n, inst.n_max)
    @assert (p > q) && (n > q) && (n_max <= n)
    @assert inst.logdet_obj + inst.geomean_obj + inst.rootdet_obj == 1
    V = randn(q, p)

    model = JuMP.Model()
    JuMP.@variable(model, np[1:p])
    JuMP.@constraint(model, vcat(n_max / 2, np .- n_max / 2) in MOI.NormInfinityCone(p + 1))
    Q = V * diagm(np) * V' # information matrix
    JuMP.@constraint(model, sum(np) == n)
    v1 = [Q[i, j] for i in 1:q for j in 1:i] # vectorized Q

    # hypograph of logdet/rootdet/geomean
    JuMP.@variable(model, hypo)
    JuMP.@objective(model, Max, hypo)

    if inst.logdet_obj
        JuMP.@constraint(model, vcat(hypo, 1.0, v1) in MOI.LogDetConeTriangle(q))
    elseif inst.rootdet_obj
        JuMP.@constraint(model, vcat(hypo, v1) in MOI.RootDetConeTriangle(q))
    else
        # hypogeomean + epinormeucl formulation
        JuMP.@variable(model, lowertri[i in 1:q, j in 1:i])
        JuMP.@variable(model, W[1:p, 1:q])
        VW = V * W
        JuMP.@constraints(model, begin
            [i in 1:q, j in 1:i], VW[i, j] == lowertri[i, j]
            [i in 1:q, j in (i + 1):q], VW[i, j] == 0
            vcat(hypo, [lowertri[i, i] for i in 1:q]) in MOI.GeometricMeanCone(q + 1)
            [i in 1:p], vcat(sqrt(q) * np[i], W[i, :]) in JuMP.SecondOrderCone()
        end)
    end

    return model
end

insts = Dict()
insts["minimal"] = [
    ((2, 3, 4, 2, true, false, false),),
    ((2, 3, 4, 2, true, false, false), StandardConeOptimizer),
    ((2, 3, 4, 2, false, true, false),),
    ((2, 3, 4, 2, false, true, false), StandardConeOptimizer),
    ((2, 3, 4, 2, false, false, true),),
    ((2, 3, 4, 2, false, false, true), StandardConeOptimizer),
    ]
insts["fast"] = [
    ((3, 5, 7, 2, true, false, false),),
    ((3, 5, 7, 2, true, false, false), StandardConeOptimizer),
    ((3, 5, 7, 2, false, true, false),),
    ((3, 5, 7, 2, false, true, false), StandardConeOptimizer),
    ((3, 5, 7, 2, false, false, true),),
    ((3, 5, 7, 2, false, false, true), StandardConeOptimizer),
    ((5, 15, 25, 5, true, false, false),),
    ((5, 15, 25, 5, false, true, false),),
    ((5, 15, 25, 5, false, false, true),),
    ((10, 30, 50, 5, true, false, false),),
    ((10, 30, 50, 5, false, true, false),),
    ((10, 30, 50, 5, false, false, true),),
    ((25, 75, 125, 10, true, false, false),),
    ((25, 75, 125, 10, false, true, false),),
    ((25, 75, 125, 10, false, false, true),),
    ]
insts["slow"] = [
    ((25, 75, 125, 10, true, false, false), StandardConeOptimizer),
    ((25, 75, 125, 10, false, true, false), StandardConeOptimizer),
    ((25, 75, 125, 10, false, false, true), StandardConeOptimizer),
    ((100, 200, 200, 10, true, false, false),),
    ((100, 200, 200, 10, false, true, false),),
    ((100, 200, 200, 10, false, false, true),),
    ]
return (ExpDesignJuMP, insts)
