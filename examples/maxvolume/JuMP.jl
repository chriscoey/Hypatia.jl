#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find maximum volume hypercube with edges parallel to the axes inside a polyhedron
defined with l_1, l_infty, or l_2 ball constraints (different to native.jl)
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))

struct MaxVolumeJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    epipernormeucl_constr::Bool # add an L2 ball constraint, else don't add
    epinorminf_constrs::Bool # add L1 and Linfty ball constraints, else don't add
end

example_tests(::Type{MaxVolumeJuMP{Float64}}, ::MinimalInstances) = [
    ((2, true, false),),
    ((2, true, true),),
    ((2, false, true),),
    ((2, false, true), ClassicConeOptimizer),
    ((2, false, true), SOConeOptimizer),
    # ((2, false, true), ExpConeOptimizer), # TODO waiting for MOI bridges geomean to exp
    ]
example_tests(::Type{MaxVolumeJuMP{Float64}}, ::FastInstances) = [
    ((10, true, false),),
    ((10, false, true),),
    ((10, false, true), ClassicConeOptimizer),
    ((10, true, true),),
    ((100, true, false),),
    ((100, false, true),),
    ((100, false, true), ClassicConeOptimizer),
    ((100, true, true),),
    ((1000, true, false),),
    ((1000, true, true),), # with bridges extended formulation will need to go into slow list
    ]
example_tests(::Type{MaxVolumeJuMP{Float64}}, ::SlowInstances) = [
    ((1000, false, true), ClassicConeOptimizer),
    ((2000, true, false),),
    ((2000, false, true),),
    ((2000, true, true),),
    ]
example_tests(::Type{MaxVolumeJuMP{Float64}}, ::ExpInstances) = [
    ((10, false, true), ClassicConeOptimizer),
    ((100, false, true), ClassicConeOptimizer),
    ((1000, true, true), ClassicConeOptimizer),
    ]

function build(inst::MaxVolumeJuMP{T}) where {T <: Float64} # TODO generic reals
    n = inst.n
    A = randn(n, n)
    # ensure there will be a feasible solution
    x = randn(n)
    gamma = norm(A * x) / sqrt(n)

    model = JuMP.Model()
    JuMP.@variable(model, t)
    JuMP.@variable(model, end_pts[1:n])
    JuMP.@objective(model, Max, t)
    JuMP.@constraint(model, vcat(t, end_pts) in MOI.GeometricMeanCone(n + 1))

    if inst.epipernormeucl_constr
        JuMP.@constraint(model, vcat(gamma, A * end_pts) in JuMP.SecondOrderCone())
    end
    if inst.epinorminf_constrs
        JuMP.@constraint(model, vcat(gamma, A * end_pts) in MOI.NormInfinityCone(n + 1))
        JuMP.@constraint(model, vcat(sqrt(n) * gamma, A * end_pts) in MOI.NormOneCone(n + 1))
    end

    return model
end

return MaxVolumeJuMP
