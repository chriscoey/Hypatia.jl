#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

see https://www.cvxpy.org/examples/dgp/pf_matrix_completion.html
modified to use spectral norm
=#

using LinearAlgebra
import Random
using Test
import Hypatia
const CO = Hypatia.Cones

function matrixcompletion(
    T::Type{<:Real},
    m::Int,
    n::Int;
    use_power::Bool = true,
    use_epinormspectral::Bool = true,
    )
    @assert m <= n

    num_known = round(Int, m * n * 0.1)
    known_rows = rand(1:m, num_known)
    known_cols = rand(1:n, num_known)
    known_vals = rand(T, num_known) .- T(0.5)

    mat_to_vec_idx(i::Int, j::Int) = (j - 1) * m + i

    is_known = fill(false, m * n)
    # h for the rows that X (the matrix and not epigraph variable) participates in
    h_norm_x = zeros(T, m * n)
    for (k, (i, j)) in enumerate(zip(known_rows, known_cols))
        known_idx = mat_to_vec_idx(i, j)
        # if not using the epinorminf cone, indices relate to X'
        h_norm_x[known_idx] = known_vals[k]
        is_known[known_idx] = true
    end

    num_known = sum(is_known) # if randomly generated, some indices may repeat
    num_unknown = m * n - num_known
    c = vcat(one(T), zeros(T, num_unknown))
    b = T[]

    # epinormspectral cone- get vec(X) in G and h
    if use_epinormspectral
        G_norm = zeros(T, m * n, num_unknown)
        total_idx = 1
        unknown_idx = 1
        for j in 1:n, i in 1:m
            if !is_known[total_idx]
                G_norm[total_idx, unknown_idx] = -1
                unknown_idx += 1
            end
            total_idx += 1
        end

        # add first row and column for epigraph variable
        G_norm = [
            -one(T)    zeros(T, 1, num_unknown);
            zeros(T, m * n)    G_norm;
            ]
        h_norm_x = vcat(zero(T), h_norm_x)
        h_norm = h_norm_x

        cones = CO.Cone{T}[CO.EpiNormSpectral{T}(m, n)]
    else
        num_rows = div(m * (m + 1), 2) + m * n + div(n * (n + 1), 2)
        G_norm = zeros(T, num_rows, num_unknown + 1)
        h_norm = zeros(T, num_rows)
        # first block epigraph variable * I
        for i in 1:m
            G_norm[sum(1:i), 1] = -1
        end
        offset = div(m * (m + 1), 2)
        # index to count rows in the bottom half of the large to-be-PSD matrix
        idx = 1
        # index only in X
        var_idx = 1
        # index of unknown vars (the x variables in the standard from), can increment it because we are moving row wise in X'
        unknown_idx = 1
        # fill bottom `n` rows
        for i in 1:n
            # X'
            for j in 1:m
                if !is_known[var_idx]
                    G_norm[offset + idx, 1 + unknown_idx] = -1
                    unknown_idx += 1
                else
                    h_norm[offset + idx] = h_norm_x[var_idx]
                end
                idx += 1
                var_idx += 1
            end
            # second block epigraph variable * I
            # skip `i` rows which will be filled with zeros
            idx += i
            G_norm[offset + idx - 1, 1] = -1
        end
        MU.vec_to_svec_cols!(G_norm, sqrt(T(2)))
        MU.vec_to_svec!(h_norm, sqrt(T(2)))
        cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(num_rows)]
    end

    if use_power
        # hypogeomean for values to be filled
        G_geo = zeros(T, num_unknown + 1, num_unknown + 1)
        total_idx = 1
        unknown_idx = 1
        for j in 1:n, i in 1:m
            if !is_known[mat_to_vec_idx(i, j)]
                G_geo[unknown_idx, unknown_idx + 1] = -1
                unknown_idx += 1
            end
            total_idx += 1
        end
        # first component of the vector in the in power cone, elements multiply to one
        h2 = vcat(zeros(T, num_unknown), one(T))
        h = vcat(h_norm, h2)
        @assert total_idx - 1 == m * n
        @assert unknown_idx - 1 == num_unknown

        A = zeros(T, 0, 1 + num_unknown)
        push!(cones, CO.Power{T}(fill(inv(T(num_unknown)), num_unknown), 1))
    else
        # number of 3-dimensional power cones needed is num_unknown - 1, number of new variables is num_unknown - 2
        # first num_unknown columns overlap with G_norm, column for the epigraph variable of the spectral cone added later
        G_geo = zeros(T, 3 * (num_unknown - 1), 2 * num_unknown - 2)
        # first cone is a special case since two of the original variables participate in it
        G_geo[1, 1] = -1
        G_geo[2, 2] = -1
        G_geo[3, num_unknown + 1] = -1
        push!(cones, CO.Power{T}(fill(inv(T(2)), 2), 1))
        offset = 4
        # loop over new vars
        for i in 1:(num_unknown - 3)
            G_geo[offset + 2, num_unknown + i + 1] = -1
            G_geo[offset + 1, num_unknown + i] = -1
            G_geo[offset, i + 2] = -1
            push!(cones, CO.Power{T}([inv(T(i + 2)), T(i + 1) / T(i + 2)], 1))
            offset += 3
        end

        # last row also special becuase hypograph variable is fixed
        G_geo[offset, num_unknown] = -1
        G_geo[offset + 1, 2 * num_unknown - 2] = -1
        push!(cones, CO.Power{T}([inv(T(num_unknown)), T(num_unknown - 1) / T(num_unknown)], 1))
        h = vcat(h_norm, zeros(T, 3 * (num_unknown - 2)), T[0, 0, 1])

        # G_norm needs to be post-padded with columns for 3dim cone vars
        G_norm = hcat(G_norm, zeros(T, size(G_norm, 1), num_unknown - 2))
        # G_geo needs to be pre-padded with the epigraph variable for the spectral norm cone
        G_geo = hcat(zeros(T, 3 * (num_unknown - 1)), G_geo)
        c = vcat(c, zeros(T, num_unknown - 2))
        A = zeros(T, 0, size(G_geo, 2))
    end
    G = vcat(G_norm, G_geo)

    return (c = c, A = A, b = b, G = G, h = h, cones = cones)
end

matrixcompletion1(T::Type{<:Real}) = matrixcompletion(T, 5, 6)
matrixcompletion2(T::Type{<:Real}) = matrixcompletion(T, 5, 6, use_power = false)
matrixcompletion3(T::Type{<:Real}) = matrixcompletion(T, 5, 6, use_epinormspectral = false)
matrixcompletion4(T::Type{<:Real}) = matrixcompletion(T, 5, 6, use_power = false, use_epinormspectral = false)
matrixcompletion5(T::Type{<:Real}) = matrixcompletion(T, 6, 8)
matrixcompletion6(T::Type{<:Real}) = matrixcompletion(T, 6, 8, use_power = false)
matrixcompletion7(T::Type{<:Real}) = matrixcompletion(T, 6, 8, use_epinormspectral = false)
matrixcompletion8(T::Type{<:Real}) = matrixcompletion(T, 6, 8, use_power = false, use_epinormspectral = false)
matrixcompletion9(T::Type{<:Real}) = matrixcompletion(T, 8, 8)
matrixcompletion10(T::Type{<:Real}) = matrixcompletion(T, 8, 8, use_power = false)
matrixcompletion11(T::Type{<:Real}) = matrixcompletion(T, 8, 8, use_epinormspectral = false)
matrixcompletion12(T::Type{<:Real}) = matrixcompletion(T, 8, 8, use_power = false, use_epinormspectral = false)

instances_matrixcompletion_all = [
    matrixcompletion1,
    matrixcompletion2,
    matrixcompletion3,
    matrixcompletion4,
    matrixcompletion5,
    matrixcompletion6,
    matrixcompletion7,
    matrixcompletion8,
    matrixcompletion9,
    matrixcompletion10,
    matrixcompletion11,
    matrixcompletion12,
    ]
instances_matrixcompletion_few = [
    matrixcompletion1,
    matrixcompletion2,
    matrixcompletion3,
    matrixcompletion4,
    ]

function test_matrixcompletion(instance::Function; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = instance(T)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @test r.status == :Optimal
    return
end
