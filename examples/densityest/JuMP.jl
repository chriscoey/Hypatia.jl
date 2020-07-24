#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

see description in native.jl
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))
import DelimitedFiles

struct DensityEstJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    dataset_name::Symbol
    X::Matrix{T}
    deg::Int
    use_wsos::Bool # use WSOS cone formulation, else PSD formulation
end
function DensityEstJuMP{Float64}(dataset_name::Symbol, deg::Int, use_wsos::Bool)
    X = DelimitedFiles.readdlm(joinpath(@__DIR__, "data", "$dataset_name.txt"))
    return DensityEstJuMP{Float64}(dataset_name, X, deg, use_wsos)
end
function DensityEstJuMP{Float64}(num_obs::Int, n::Int, args...)
    X = randn(num_obs, n)
    return DensityEstJuMP{Float64}(:Random, X, args...)
end

function build(inst::DensityEstJuMP{T}) where {T <: Float64} # TODO generic reals
    X = inst.X
    (num_obs, n) = size(X)
    domain = ModelUtilities.Box{Float64}(-ones(n), ones(n)) # domain is unit box [-1,1]^n

    # rescale X to be in unit box
    minX = minimum(X, dims = 1)
    maxX = maximum(X, dims = 1)
    X .-= (minX + maxX) / 2
    X ./= (maxX - minX) / 2

    # setup interpolation
    halfdeg = div(inst.deg + 1, 2)
    (U, _, Ps, V, w) = ModelUtilities.interpolate(domain, halfdeg, calc_V = true, calc_w = true)
    # TODO maybe incorporate this interp-basis transform into MU, and do something smarter for uni/bi-variate
    F = qr!(Array(V'), Val(true))
    V_X = ModelUtilities.make_chebyshev_vandermonde(X, 2halfdeg)
    X_pts_polys = F \ V_X'

    model = JuMP.Model()
    JuMP.@variable(model, z)
    JuMP.@objective(model, Max, z)
    JuMP.@variable(model, f_pts[1:U])

    # objective epigraph
    JuMP.@constraint(model, vcat(z, X_pts_polys' * f_pts) in MOI.GeometricMeanCone(1 + num_obs))

    # density integrates to 1
    JuMP.@constraint(model, dot(w, f_pts) == 1)

    # density nonnegative
    if inst.use_wsos
        # WSOS formulation
        JuMP.@constraint(model, f_pts in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps))
    else
        # PSD formulation
        psd_vars = []
        for (r, Pr) in enumerate(Ps)
            Lr = size(Pr, 2)
            psd_r = JuMP.@variable(model, [1:Lr, 1:Lr], Symmetric)
            push!(psd_vars, psd_r)
            JuMP.@SDconstraint(model, psd_r >= 0)
        end
        coeffs_lhs = JuMP.@expression(model, [u in 1:U], sum(sum(Pr[u, k] * Pr[u, l] * psd_r[k, l] * (k == l ? 1 : 2) for k in 1:size(Pr, 2) for l in 1:k) for (Pr, psd_r) in zip(Ps, psd_vars)))
        JuMP.@constraint(model, coeffs_lhs .== f_pts)
    end

    return model
end

instances[DensityEstJuMP]["minimal"] = [
    ((5, 1, 2, true),),
    ((:iris, 2, true),),
    ]
instances[DensityEstJuMP]["fast"] = [
    ((10, 1, 5, true),),
    ((10, 1, 10, true),),
    ((100, 1, 20, true),),
    ((100, 1, 50, true),),
    ((100, 1, 100, true),),
    ((200, 1, 200, true),),
    ((500, 1, 500, true),),
    ((100, 2, 5, true),),
    ((100, 2, 10, true),),
    ((200, 2, 20, true),),
    ((50, 3, 2, true),),
    ((50, 3, 4, true),),
    ((500, 3, 14, true),),
    ((50, 4, 2, true),),
    ((100, 8, 2, true),),
    ((100, 8, 2, false),),
    ((250, 4, 4, true),),
    ((250, 4, 4, false),),
    ((200, 32, 2, false),),
    ((:iris, 4, true),),
    ((:iris, 5, true),),
    ((:iris, 6, true),),
    ((:iris, 4, false),),
    ((:cancer, 4, true),),
    ]
instances[DensityEstJuMP]["slow"] = [
    ((500, 2, 60, true),),
    ((1000, 3, 20, true),),
    ((200, 4, 4, false),),
    ((500, 4, 6, true),),
    ((500, 4, 6, false),),
    ]
