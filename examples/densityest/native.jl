#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

given a sequence of observations X_1,...,X_n with each Xᵢ in Rᵈ,
find a density function f maximizing the log likelihood of the observations
    min -∑ᵢ zᵢ
    -zᵢ + log(f(Xᵢ)) ≥ 0 ∀ i = 1,...,n
    ∫f = 1
    f ≥ 0
==#

using LinearAlgebra
import Random
using Test
import Hypatia
import Hypatia.BlockMatrix
const CO = Hypatia.Cones
const MU = Hypatia.ModelUtilities

include(joinpath(@__DIR__, "data.jl"))

function densityest(
    T::Type{<:Real},
    X::Matrix{Float64},
    deg::Int;
    use_sumlog::Bool = false,
    use_wsos::Bool = true,
    sample_factor::Int = 100,
    use_linops::Bool = false,
    )
    (nobs, dim) = size(X)
    X = convert(Matrix{T}, X)

    domain = MU.Box{T}(-ones(T, dim), ones(T, dim))
    # rescale X to be in unit box
    minX = minimum(X, dims = 1)
    maxX = maximum(X, dims = 1)
    X .-= (minX + maxX) / 2
    X ./= (maxX - minX) / 2

    halfdeg = div(deg + 1, 2)
    (U, pts, P0, PWts, w) = MU.interpolate(domain, halfdeg, sample = true, calc_w = true, sample_factor = sample_factor)
    lagrange_polys = MU.recover_lagrange_polys(pts, 2 * halfdeg)
    basis_evals = Matrix{T}(undef, nobs, U)
    for i in 1:nobs, j in 1:U
        basis_evals[i, j] = lagrange_polys[j](X[i, :])
    end
    # TODO remove below conversions when ModelUtilities can use T <: Real
    P0 = T.(P0)
    PWts = convert.(Matrix{T}, PWts)

    cones = CO.Cone{T}[]
    cone_offset = 1

    if use_wsos
        # U variables
        h_poly = zeros(T, U)
        b_poly = T[]
        push!(cones, CO.WSOSPolyInterp{T, T}(U, [P0, PWts...]))
        cone_offset += U
        num_psd_vars = 0
    else
        # U polynomial coefficient variables plus PSD variables
        # there are n new PSD variables, we will store them scaled, lower triangle, row-wise
        n = length(PWts) + 1
        psd_var_list = Matrix{T}[]
        L = size(P0, 2)
        dim = div(L * (L + 1), 2)
        num_psd_vars = dim
        # first part of A
        push!(psd_var_list, zeros(T, U, dim))
        idx = 1
        for k in 1:L, l in 1:k
            # off diagonals are doubled
            psd_var_list[1][:, idx] = P0[:, k] .* P0[:, l] * (k == l ? 1 : 2)
            idx += 1
        end
        push!(cones, CO.PosSemidefTri{T, T}(dim))
        cone_offset += dim

        for i in 1:(n - 1)
            L = size(PWts[i], 2)
            dim = div(L * (L + 1), 2)
            num_psd_vars += dim
            push!(psd_var_list, zeros(T, U, dim))
            idx = 1
            for k in 1:L, l in 1:k
                # off diagonals are doubled
                psd_var_list[i + 1][:, idx] = PWts[i][:, k] .* PWts[i][:, l] * (k == l ? 1 : 2)
                idx += 1
            end
            push!(cones, CO.PosSemidefTri{T, T}(dim))
            cone_offset += dim
        end
        A_psd = hcat(psd_var_list...)
        b_poly = zeros(T, U)
        h_poly = zeros(T, num_psd_vars)
    end

    if use_sumlog
        h_log = zeros(T, nobs + 2)
        h_log[2] = 1
        G_log = zeros(T, 2 + nobs, 1 + U)
        G_log[1, 1] = -1
        for i in 1:nobs
            G_log[i + 2, 2:(1 + U)] = -basis_evals[i, :]
        end
        push!(cones, CO.HypoPerSumLog{T}(nobs + 2))
        num_hypo_vars = 1
    else
        h_log = zeros(T, 3 * nobs)
        G_log = zeros(T, 3 * nobs, nobs + U)
        offset = 1
        for i in 1:nobs
            G_log[offset, i] = -1.0
            G_log[offset + 2, (nobs + 1):(nobs + U)] = -basis_evals[i, :]
            h_log[offset + 1] = 1.0
            offset += 3
            push!(cones, CO.HypoPerSumLog{T}(3)) # TODO use EpiPerExp (swap order of variables)
            cone_offset += 3
        end
        num_hypo_vars = nobs
    end

    (log_rows, log_cols) = size(G_log)
    if use_linops
        if use_wsos
            A = BlockMatrix{T}(
                1,
                num_hypo_vars + U,
                [T.(w')],
                [1:1],
                [(num_hypo_vars + 1):(num_hypo_vars + U)]
            )
            G = BlockMatrix{T}(
                U + log_rows,
                num_hypo_vars + U,
                [-I, G_log],
                [1:U, (U + 1):(U + log_rows)],
                [(num_hypo_vars + 1):(num_hypo_vars + U), 1:(num_hypo_vars + U)],
            )
        else
            A = BlockMatrix{T}(
                U + 1,
                num_hypo_vars + U + num_psd_vars,
                [-I, A_psd, T.(w')],
                [1:U, 1:U, (U + 1):(U + 1)],
                [(num_hypo_vars + 1):(num_hypo_vars + U), (num_hypo_vars + U + 1):(num_hypo_vars + U + num_psd_vars), (num_hypo_vars + 1):(num_hypo_vars + U)],
            )
            I_scaled = MU.vec_to_svec_cols!(Diagonal(I * one(T), num_psd_vars), sqrt(T(2)))
            G = BlockMatrix{T}(
                num_psd_vars + log_rows,
                num_hypo_vars + U + num_psd_vars,
                [-I_scaled, G_log],
                [1:num_psd_vars, (num_psd_vars + 1):(num_psd_vars + log_rows)],
                [(num_hypo_vars + U + 1):(num_hypo_vars + U + num_psd_vars), 1:(num_hypo_vars + U)],
            )
        end
    else
        if use_wsos
            A = [
                zeros(T, 1, num_hypo_vars)    T.(w');
                ]
            G = [
                zeros(T, U, num_hypo_vars)    Matrix{T}(-I, U, U);
                G_log;
                ]
        else
            A = [
                zeros(T, U, num_hypo_vars)    Matrix{T}(-I, U, U)    A_psd;
                zeros(T, 1, num_hypo_vars)    T.(w')    zeros(T, 1, num_psd_vars);
                ]
            I_scaled = MU.vec_to_svec_cols!(Matrix{T}(I, num_psd_vars, num_psd_vars), sqrt(T(2)))
            G = [
                zeros(T, num_psd_vars, num_hypo_vars + U)   -I_scaled;
                G_log   zeros(T, log_rows, num_psd_vars);
                ]
        end
    end
    h = vcat(h_poly, h_log)
    c = vcat(-ones(T, num_hypo_vars), zeros(T, U + num_psd_vars))
    b = vcat(b_poly, one(T))

    return (c = c, A = A, b = b, G = G, h = h, cones = cones)
end

densityest(T::Type{<:Real}, nobs::Int, n::Int, deg::Int; options...) = densityest(T, randn(nobs, n), deg; options...)

densityest1(T::Type{<:Real}) = densityest(T, iris_data(), 4, use_sumlog = true)
densityest2(T::Type{<:Real}) = densityest(T, iris_data(), 4, use_sumlog = false)
densityest3(T::Type{<:Real}) = densityest(T, cancer_data(), 4, use_sumlog = true)
densityest4(T::Type{<:Real}) = densityest(T, cancer_data(), 4, use_sumlog = false)
densityest5(T::Type{<:Real}) = densityest(T, 50, 1, 4, use_sumlog = true)
densityest6(T::Type{<:Real}) = densityest(T, 50, 1, 4, use_sumlog = false)
densityest7(T::Type{<:Real}) = densityest(T, 50, 1, 4, use_sumlog = true, use_wsos = false)
densityest8(T::Type{<:Real}) = densityest(T, 50, 1, 4, use_sumlog = false, use_wsos = false)
densityest9(T::Type{<:Real}) = densityest(T, 10, 1, 2, use_sumlog = true, use_wsos = true, use_linops = true)
densityest10(T::Type{<:Real}) = densityest(T, 10, 1, 2, use_sumlog = true, use_wsos = false, use_linops = true)
densityest11(T::Type{<:Real}) = densityest(T, 10, 1, 2, use_sumlog = false, use_wsos = true, use_linops = true)
densityest12(T::Type{<:Real}) = densityest(T, 10, 1, 2, use_sumlog = false, use_wsos = false, use_linops = true)

instances_densityest_all = [
    densityest1,
    densityest2,
    densityest3,
    densityest4,
    densityest5,
    densityest6,
    densityest7,
    densityest8,
    ]
instances_densityest_linops = [
    densityest9,
    densityest10,
    densityest11,
    densityest12,
    ]
instances_densityest_few = [
    densityest1,
    densityest2,
    densityest3,
    densityest4,
    densityest5,
    densityest6,
    densityest7,
    densityest8,
    ]

function test_densityest(instance::Function; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = instance(T)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @test r.status == :Optimal
    return
end
