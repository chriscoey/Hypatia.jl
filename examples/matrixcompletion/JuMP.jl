#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

see description in native.jl
=#

using SparseArrays

struct MatrixCompletionJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    num_rows::Int
    num_cols::Int
    nuclearnorm_obj::Bool # use nuclearnorm in the objective, else spectral norm
end

function build(inst::MatrixCompletionJuMP{T}) where {T <: Float64} # TODO generic reals
    (num_rows, num_cols) = (inst.num_rows, inst.num_cols)
    @assert num_rows <= num_cols
    A = randn(num_rows, num_cols)
    (row_idxs, col_idxs, _) = findnz(sprand(Bool, num_rows, num_cols, 0.1))

    model = JuMP.Model()
    JuMP.@variable(model, X[1:num_rows, 1:num_cols])
    JuMP.@variable(model, t)
    JuMP.@objective(model, Min, t)
    cone = (inst.nuclearnorm_obj ? MOI.NormNuclearCone : MOI.NormSpectralCone)
    JuMP.@constraint(model, vcat(t, vec(X)) in cone(num_rows, num_cols))
    JuMP.@constraint(model, [(row, col) in zip(row_idxs, col_idxs)], X[row, col] == A[row, col])

    return model
end

instances[MatrixCompletionJuMP]["minimal"] = [
    ((2, 3, true),),
    ((2, 3, true), ClassicConeOptimizer),
    ((2, 3, false),),
    ((2, 3, false), ClassicConeOptimizer),
    ]
instances[MatrixCompletionJuMP]["fast"] = [
    ((5, 8, true),),
    ((5, 8, true), ClassicConeOptimizer),
    ((5, 8, false),),
    ((5, 8, false), ClassicConeOptimizer),
    ((12, 24, true),),
    ((12, 24, false),),
    ((14, 140, false),),
    ]
instances[MatrixCompletionJuMP]["slow"] = [
    ((14, 140, false), ClassicConeOptimizer),
    ((14, 140, true),),
    ((14, 140, true), ClassicConeOptimizer),
    ((40, 70, true),),
    ((40, 70, false),),
    ((18, 180, true),),
    ((18, 180, false),),
    ]

# benchmark 1 instances
instances[MatrixCompletionJuMP]["bench1"] = (
    ((d1, d2, false), ext)
    for d1 in vcat(3, 10:5:55) # includes compile run
    for d2 in (5d1, 10d1)
    for ext in (nothing, ClassicConeOptimizer)
    )
