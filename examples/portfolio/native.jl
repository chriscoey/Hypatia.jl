#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

maximize expected returns subject to risk constraints
=#

using LinearAlgebra
import Random
using Test
import Hypatia
import Hypatia.HypReal
const CO = Hypatia.Cones

function portfolio(
    T::Type{<:HypReal},
    num_stocks::Int,
    risk_measures::Vector{Symbol};
    use_l1ball::Bool = true,
    use_linfball::Bool = true,
    use_linops::Bool = false,
    )
    if :entropic in risk_measures && length(risk_measures) > 1
        error("if using entropic ball, cannot specify other risk measures")
    end

    last_idx(a::Vector{UnitRange{Int}}) = a[end][end]

    returns = rand(T, num_stocks)
    sigma_half = T.(randn(num_stocks, num_stocks))
    x = T.(randn(num_stocks))
    x ./= norm(x)
    gamma = sum(abs, sigma_half * x) / sqrt(T(num_stocks))

    c = returns
    # investments add to one, nonnegativity
    if use_linops
        A_blocks = Any[ones(T, 1, num_stocks)]
        A_rows = [1:1]
        A_cols = [1:num_stocks]
        G_blocks = Any[-I]
        G_rows = [1:num_stocks]
        G_cols = [1:num_stocks]
    else
        A = ones(T, 1, num_stocks)
        G = Matrix{T}(-I, num_stocks, num_stocks)
    end
    b = T[1]
    h = zeros(T, num_stocks)
    cone_idxs = UnitRange{Int}[1:num_stocks]
    cones = CO.Cone{T}[CO.Nonnegative{T}(num_stocks)]
    cone_offset = num_stocks

    function add_single_ball(cone, gamma_new)
        if use_linops
            push!(G_blocks, -sigma_half)
            push!(G_rows, (cone_offset + 2):(cone_offset + num_stocks + 1))
            push!(G_cols, 1:num_stocks)
        else
            G = vcat(G, zeros(T, 1, num_stocks), -sigma_half)
        end
        h_risk = vcat(gamma_new, zeros(T, num_stocks))
        h = vcat(h, h_risk)
        push!(cones, cone)
        push!(cone_idxs, (cone_offset + 1):(cone_offset + num_stocks + 1))
        cone_offset += num_stocks + 1
    end

    if :quadratic in risk_measures
        add_single_ball(CO.EpiNormEucl{T}(num_stocks + 1), gamma)
    end
    if :l1 in risk_measures && use_l1ball
        add_single_ball(CO.EpiNormInf{T}(num_stocks + 1, true), gamma * sqrt(T(num_stocks)))
    end
    if :linf in risk_measures && use_linfball
        add_single_ball(CO.EpiNormInf{T}(num_stocks + 1), gamma)
    end

    if :l1 in risk_measures && !use_l1ball
        c = vcat(c, zeros(T, 2 * num_stocks))
        if use_linops
            push!(A_blocks, sigma_half)
            push!(A_blocks, -I)
            push!(A_blocks, I)
            push!(A_blocks, ones(T, 1, 2 * num_stocks))

            A_offset = last_idx(A_rows)
            append!(A_rows, fill((A_offset + 1):(A_offset + num_stocks), 3))
            push!(A_rows, (last_idx(A_rows) + 1):(last_idx(A_rows) + 1))

            push!(A_cols, 1:num_stocks)
            push!(A_cols, (num_stocks + 1):(2 * num_stocks))
            push!(A_cols, (2 * num_stocks + 1):(3 * num_stocks))
            push!(A_cols, (num_stocks + 1):(3 * num_stocks))

            push!(G_blocks, -I)
            push!(G_rows, (last_idx(G_rows) + 1):(last_idx(G_rows) + 2 * num_stocks))
            push!(G_cols, (last_idx(G_cols) + 1):(last_idx(G_cols) + 2 * num_stocks))
        else
            id = Matrix{T}(I, num_stocks, num_stocks)
            id2 = Matrix{T}(I, 2 * num_stocks, 2 * num_stocks)
            A = [
                A    zeros(T, 1, 2 * num_stocks);
                sigma_half    -id    id;
                zeros(T, 1, num_stocks)    ones(T, 1, 2 * num_stocks);
                ]
            G = [
                G    zeros(T, size(G, 1), 2 * num_stocks);
                zeros(T, 2 * num_stocks, num_stocks)    -id2;
                ]
        end
        b = vcat(b, zeros(T, num_stocks), gamma * sqrt(T(num_stocks)))
        h = vcat(h, zeros(T, 2 * num_stocks))
        push!(cones, CO.Nonnegative{T}(2 * num_stocks))
        push!(cone_idxs, (cone_offset + 1):(cone_offset + 2 * num_stocks))
        cone_offset += 2 * num_stocks
    end

    if :linf in risk_measures && !use_linfball
        if use_linops
            push!(G_blocks, sigma_half)
            push!(G_blocks, -sigma_half)
            push!(G_rows, (cone_offset + 1):(cone_offset + num_stocks))
            push!(G_rows, (cone_offset + num_stocks + 1):(cone_offset + 2 * num_stocks))
            push!(G_cols, 1:num_stocks)
            push!(G_cols, 1:num_stocks)
        else
            padding = zeros(T, num_stocks, size(G, 2) - num_stocks)
            G = [
                G;
                sigma_half    padding;
                -sigma_half    padding;
                ]
        end
        h = vcat(h, gamma * ones(T, 2 * num_stocks))
        push!(cones, CO.Nonnegative{T}(2 * num_stocks))
        push!(cone_idxs, (cone_offset + 1):(cone_offset + 2 * num_stocks))
        cone_offset += 2 * num_stocks
    end

    if :entropic in risk_measures
        # sigma_half = abs.(sigma_half) TODO will this always be feasible?
        c = vcat(c, zeros(T, 2 * num_stocks))
        b = vcat(b, gamma^2)
        col_offset = (use_linops ? last_idx(G_cols) : size(G, 2))
        G2pos = zeros(T, 3 * num_stocks, 2 * num_stocks + col_offset)
        G2neg = copy(G2pos)
        h2 = zeros(T, 3 * num_stocks)

        row_offset = 1
        for i in 1:num_stocks
            G2pos[row_offset, num_stocks + i] = 1 # entropy
            G2pos[row_offset + 1, 1:num_stocks] = -sigma_half[i, :]
            h2[row_offset + 1] = 1
            h2[row_offset + 2] = 1
            G2neg[row_offset, 2 * num_stocks + i] = 1 # entropy
            G2neg[row_offset + 1, 1:num_stocks] = sigma_half[i, :]
            row_offset += 3
        end

        if use_linops
            push!(A_blocks, ones(T, 1, 2 * num_stocks))
            push!(A_rows, (last_idx(A_rows) + 1):(last_idx(A_rows) + 1))
            push!(A_cols, (last_idx(A_cols) + 1):(last_idx(A_cols) + 2 * num_stocks))
            push!(G_blocks, vcat(G2pos, G2neg))
            push!(G_rows, (cone_offset + 1):(cone_offset + 6 * num_stocks))
            push!(G_cols, 1:(2 * num_stocks + col_offset))
        else
            A = [
                A    zeros(T, size(A, 1), 2 * num_stocks);
                zeros(T, 1, num_stocks)    ones(T, 1, 2 * num_stocks);
                ]
            G = [
                G    zeros(T, size(G, 1), 2 * num_stocks);
                G2pos;
                G2neg;
                ]
        end
        h = vcat(h, h2, h2)
        for i in 1:(2 * num_stocks)
            push!(cone_idxs, (3 * (i - 1) + cone_offset + 1):(3 * i + cone_offset))
            push!(cones, CO.HypoPerLog{T}(3))
        end
        cone_offset += 6 * num_stocks
    end

    if use_linops
        A = Hypatia.HypBlockMatrix{T}(last_idx(A_rows), last_idx(A_cols), A_blocks, A_rows, A_cols)
        G = Hypatia.HypBlockMatrix{T}(last_idx(G_rows), last_idx(G_cols), G_blocks, G_rows, G_cols)
    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

portfolio1(T::Type{<:HypReal}) = portfolio(T, 4, [:l1], use_l1ball = true)
portfolio2(T::Type{<:HypReal}) = portfolio(T, 6, [:l1], use_l1ball = false)
portfolio3(T::Type{<:HypReal}) = portfolio(T, 4, [:linf], use_linfball = true)
portfolio4(T::Type{<:HypReal}) = portfolio(T, 6, [:linf], use_linfball = false)
portfolio5(T::Type{<:HypReal}) = portfolio(T, 4, [:linf, :l1], use_linfball = true, use_l1ball = true)
portfolio6(T::Type{<:HypReal}) = portfolio(T, 6, [:linf, :l1], use_linfball = false, use_l1ball = false)
portfolio7(T::Type{<:HypReal}) = portfolio(T, 4, [:linf, :l1], use_l1ball = true, use_linops = false)
portfolio8(T::Type{<:HypReal}) = portfolio(T, 4, [:linf, :l1], use_l1ball = true, use_linops = true)
portfolio9(T::Type{<:HypReal}) = portfolio(T, 3, [:linf, :l1], use_l1ball = false, use_linops = false)
portfolio10(T::Type{<:HypReal}) = portfolio(T, 3, [:linf, :l1], use_l1ball = false, use_linops = true)
portfolio11(T::Type{<:HypReal}) = portfolio(T, 4, [:entropic], use_l1ball = false, use_linops = false)
portfolio12(T::Type{<:HypReal}) = portfolio(T, 4, [:entropic], use_l1ball = false, use_linops = true)

instances_portfolio_all = [
    portfolio1,
    portfolio2,
    portfolio3,
    portfolio4,
    portfolio5,
    portfolio6,
    portfolio7,
    portfolio9,
    portfolio11,
    ]
instances_portfolio_few = [
    portfolio1,
    portfolio3,
    portfolio5,
    portfolio6,
    ]
instances_portfolio_linops = [
    portfolio8,
    portfolio10,
    portfolio12,
    ]

function test_portfolio(instance::Function; T::Type{<:HypReal} = Float64, test_options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    d = instance(T)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs; test_options..., atol = tol, rtol = tol)
    @test r.status == :Optimal
    return
end
