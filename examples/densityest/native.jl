#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

given a sequence of observations X₁,...,Xᵢ with each Xᵢ in Rᵈ,
find a density function f maximizing the log likelihood of the observations
(equivalent to maximizing geomean of f.(X))
    max (prod_i f(Xᵢ))^(1/n)
    ∫f = 1
    f ≥ 0
=#

include(joinpath(@__DIR__, "../common_native.jl"))
import DelimitedFiles

struct DensityEstNative{T <: Real} <: ExampleInstanceNative{T}
    dataset_name::Symbol
    X::Matrix{T}
    deg::Int
    use_wsos::Bool # use WSOS cone formulation, else PSD formulation
    hypogeomean_obj::Bool # use geomean objective, else sum of logs objective
    use_hypogeomean::Bool # use hypogeomean cone if applicable, else 3-dim entropy formulation
end
function DensityEstNative{T}(
    dataset_name::Symbol,
    deg::Int,
    use_wsos::Bool,
    hypogeomean_obj::Bool,
    use_hypogeomean::Bool,
    ) where {T <: Real}
    X = DelimitedFiles.readdlm(joinpath(@__DIR__, "data", "$dataset_name.txt"))
    X = convert(Matrix{T}, X)
    return DensityEstNative{T}(dataset_name, X, deg, use_wsos, hypogeomean_obj, use_hypogeomean)
end
function DensityEstNative{T}(num_obs::Int, n::Int, args...) where {T <: Real}
    X = randn(T, num_obs, n)
    return DensityEstNative{T}(:Random, X, args...)
end

example_tests(::Type{<:DensityEstNative{<:BlasReal}}, ::MinimalInstances) = [
    ((5, 1, 2, true, true, true),),
    ((5, 1, 2, false, true, true),),
    ((5, 2, 1, false, true, true),),
    ((5, 1, 2, true, false, true),),
    ((5, 1, 2, true, true, false),),
    ((:iris, 2, true, true, true),),
    ]
example_tests(::Type{DensityEstNative{Float64}}, ::FastInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    return [
    ((50, 1, 4, true, true, true), options),
    ((50, 1, 10, true, true, true), options),
    ((50, 1, 50, true, true, true), options),
    ((100, 1, 100, true, true, true), options),
    ((500, 1, 500, true, true, true), options),
    ((50, 2, 2, true, true, true), options),
    ((200, 2, 20, true, true, true), options),
    ((50, 2, 2, false, true, true), options),
    ((50, 2, 2, true, false, true), options),
    ((50, 2, 2, true, true, false), options),
    ((500, 3, 14, true, true, true), options),
    ((20, 4, 3, false, true, false),),
    ((20, 4, 3, true, true, true),),
    ((100, 8, 2, true, true, true), options),
    ((100, 8, 2, false, true, true), options),
    ((100, 8, 2, true, false, true), options),
    ((100, 8, 2, true, true, false), options),
    ((250, 4, 6, true, true, true), options),
    ((250, 4, 6, false, true, true), options),
    ((250, 4, 6, true, false, true), options),
    ((250, 4, 6, true, true, false), options),
    ((200, 32, 2, true, true, true), options),
    ((:iris, 4, true, true, true), options),
    ((:iris, 5, true, true, true), options),
    ((:iris, 6, true, true, true), options),
    ((:iris, 4, false, true, true), options),
    ((:iris, 4, true, false, true), options),
    ((:iris, 4, true, true, false), options),
    ((:cancer, 4, true, true, true), options),
    ((:cancer, 4, false, true, true), options),
    ((:cancer, 4, true, false, true), options),
    ((:cancer, 4, true, true, false), options),
    ]
end
example_tests(::Type{DensityEstNative{Float64}}, ::SlowInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    return [
    ((:cancer, 6, true, true, true), options),
    # ((:cancer, 6, false, true, true), options),
    ((:cancer, 6, true, false, true), options),
    ((:cancer, 6, true, true, false), options),
    ((500, 2, 60, true, true, true), options),
    ((1000, 3, 20, true, true, true), options),
    ((400, 5, 6, true, true, true), options),
    ((400, 5, 6, false, true, true), options),
    ((400, 5, 6, true, false, true), options),
    ((400, 5, 6, true, true, false), options),
    ]
end
example_tests(::Type{DensityEstNative{Float64}}, ::ExpInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    return [
    ((50, 2, 2, true, true, false), options),
    ((20, 4, 3, false, true, false),),
    ((100, 8, 2, true, true, false), options),
    ((250, 4, 6, true, true, false), options),
    ((400, 5, 6, true, true, false), options),
    ]
end

function build(inst::DensityEstNative{T}) where {T <: Real}
    X = inst.X
    (num_obs, n) = size(X)
    domain = ModelUtilities.Box{T}(-ones(T, n), ones(T, n)) # domain is unit box [-1,1]^n

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
    X_pts_polys = (F \ V_X')'

    cones = Cones.Cone{T}[]

    num_psd_vars = 0
    if inst.use_wsos
        # U variables
        h_poly = zeros(T, U)
        b_poly = T[]
        push!(cones, Cones.WSOSInterpNonnegative{T, T}(U, Ps))
    else
        # U polynomial coefficient variables plus PSD variables
        # there are length(Ps) new PSD variables, we will store them scaled, lower triangle, row-wise
        psd_var_list = Matrix{T}[]
        nonneg_cone_size = 0
        for i in eachindex(Ps)
            L = size(Ps[i], 2)
            dim = Cones.svec_length(L)
            if dim == 1
                nonneg_cone_size += 1
            else
                if nonneg_cone_size > 0
                    push!(cones, Cones.Nonnegative{T}(nonneg_cone_size))
                end
                push!(cones, Cones.PosSemidefTri{T, T}(dim))
            end
            num_psd_vars += dim
            push!(psd_var_list, zeros(T, U, dim))
            idx = 1
            # we will work with PSD vars with scaled off-diagonals
            # relevant columns (not rows) in A need to be scaled by sqrt(2) also
            for k in 1:L
                for l in 1:(k - 1)
                    psd_var_list[i][:, idx] = Ps[i][:, k] .* Ps[i][:, l] * sqrt(T(2))
                    idx += 1
                end
                psd_var_list[i][:, idx] = Ps[i][:, k] .* Ps[i][:, k]
                idx += 1
            end
        end
        if nonneg_cone_size > 0
            push!(cones, Cones.Nonnegative{T}(nonneg_cone_size))
        end
        A_psd = hcat(psd_var_list...)
        b_poly = zeros(T, U)
        h_poly = zeros(T, num_psd_vars)
    end

    if inst.hypogeomean_obj
        num_hypo_vars = 1
        if inst.use_hypogeomean
            G_likl = [
                -one(T) zeros(T, 1, U)
                zeros(T, num_obs) -X_pts_polys
                ]
            h_likl = zeros(T, 1 + num_obs)
            push!(cones, Cones.HypoGeomean{T}(fill(inv(T(num_obs)), num_obs)))
            A_ext = zeros(T, 0, num_obs)
        else
            num_ext_geom_vars = 1 + num_obs
            ext_offset = 2 + U + num_psd_vars
            h_likl = zeros(T, 3 * num_obs + 2)
            # order of variables: hypograph u, U f(obs) vars, psd vars, geomean ext vars (y, z)
            G_likl = zeros(T, 3 * num_obs + 2, ext_offset + num_obs)
            # y >= u, e'z >= 0
            G_likl[1, 1] = 1
            G_likl[1, ext_offset] = -1
            G_likl[2, (end - num_obs + 1):end] .= -1
            push!(cones, Cones.Nonnegative{T}(2))
            # f(x) <= y * log(z / y)
            row_offset = 3
            for i in 1:num_obs
                G_likl[row_offset, ext_offset + i] = 1
                G_likl[row_offset + 1, 2:(1 + U)] = -X_pts_polys[i, :]
                G_likl[row_offset + 2, ext_offset] = -1
                row_offset += 3
                push!(cones, Cones.EpiSumPerEntropy{T}(3))
            end
        end
    else
        num_hypo_vars = num_obs
        h_likl = zeros(T, 3 * num_obs)
        G_likl = zeros(T, 3 * num_obs, num_obs + U)
        offset = 1
        for i in 1:num_obs
            G_likl[offset + 2, (num_obs + 1):(num_obs + U)] = -X_pts_polys[i, :]
            h_likl[offset + 1] = 1
            G_likl[offset, i] = -1
            offset += 3
            push!(cones, Cones.HypoPerLog{T}(3))
        end
    end

    # extended formulation variables for hypogeomean cone are added after psd ones, so psd vars were already accounted for in hypogeomean_obj && !use_hypogeomean path
    if !inst.hypogeomean_obj || inst.use_hypogeomean
        G_likl = hcat(G_likl, zeros(T, size(G_likl, 1), num_psd_vars))
        num_ext_geom_vars = 0
    end
    c = zeros(T, num_hypo_vars + U + num_psd_vars + num_ext_geom_vars)
    @views c[1:num_hypo_vars] .= -1
    h = vcat(h_poly, h_likl)
    b = vcat(b_poly, one(T))

    if inst.use_wsos
        A = zeros(T, 1, num_hypo_vars + U + num_ext_geom_vars)
        A[1, num_hypo_vars .+ (1:U)] = w
        G = zeros(T, U + size(G_likl, 1), size(G_likl, 2))
        G[1:U, num_hypo_vars .+ (1:U)] = Diagonal(-I, U)
        G[(U + 1):end, :] = G_likl
    else
        A = zeros(T, U + 1, num_hypo_vars + U + num_psd_vars + num_ext_geom_vars)
        A[1:U, num_hypo_vars .+ (1:U)] = Diagonal(-I, U)
        A[1:U, (num_hypo_vars + U) .+ (1:num_psd_vars)] = A_psd
        A[U + 1, num_hypo_vars .+ (1:U)] = w
        G = zeros(T, num_psd_vars + size(G_likl, 1), size(G_likl, 2))
        G[1:num_psd_vars, (num_hypo_vars + U) .+ (1:num_psd_vars)] = Diagonal(-I, num_psd_vars)
        G[(num_psd_vars + 1):end, :] = G_likl
    end

    model = Models.Model{T}(c, A, b, G, h, cones)
    return model
end

return DensityEstNative
