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
import DynamicPolynomials
import Hypatia
const HYP = Hypatia
const MO = HYP.Models
const MU = HYP.ModelUtilities
const CO = HYP.Cones
const SO = HYP.Solvers

THR = Type{<: HYP.HypReal}

# include(joinpath(@__DIR__, "data.jl"))

function densityest(
    X::AbstractMatrix{Float64},
    deg::Int;
    use_sumlog::Bool = false,
    use_wsos::Bool = true,
    sample_factor::Int = 100,
    T::THR = Float64,
    alpha = -1,
    )
    (nobs, dim) = size(X)

    domain = MU.Box(-ones(dim), ones(dim))
    # rescale X to be in unit box
    minX = minimum(X, dims = 1)
    maxX = maximum(X, dims = 1)
    X .-= (minX + maxX) / 2
    X ./= (maxX - minX) / 2

    halfdeg = div(deg + 1, 2)
    (U, pts, P0, PWts, w) = MU.interpolate(domain, halfdeg, sample = true, calc_w = true, sample_factor = sample_factor)

    lagrange_polys = MU.recover_lagrange_polys(pts, 2 * halfdeg)
    basis_evals = Matrix{Float64}(undef, nobs, U)
    for i in 1:nobs, j in 1:U
        basis_evals[i, j] = lagrange_polys[j](X[i, :])
    end

    P0c = convert.(T, P0)
    PWtsc = Matrix{T}[]
    for pwt in PWts
        push!(PWtsc, convert.(T, pwt))
    end

    cones = CO.Cone[]
    cone_idxs = UnitRange{Int}[]
    cone_offset = 1
    if use_wsos
        # will set up for U variables
        G_poly = -Matrix{T}(I, U, U)
        h_poly = zeros(T, U)
        c_poly = zeros(T, U)
        A_poly = zeros(T, 0, U)
        b_poly = T[]
        c_poly = T[]
        push!(cones, CO.WSOSPolyInterp{T, T}(U, [P0c, PWtsc...]))
        push!(cone_idxs, 1:U)
        cone_offset += U
    else
        # will set up for U coefficient variables plus PSD variables
        # there are n new PSD variables, we will store them scaled, lower triangle, row-wise
        n = length(PWts) + 1
        psd_vars = 0
        a_poly = Vector{Matrix{T}}(n)
        L = size(P0, 2)
        dim = div(L * (L + 1), 2)
        num_poly_vars += dim
        # first part of A
        push!(a_poly, zeros(T, U, dim))
        idx = 1
        for k in 1:L, l in 1:k
            # off diagonals are doubled, but already scaled by rt2
            a_poly[1][:, idx] = P0c[:, k] .* P0c[:, l] * (k == l ? 1 : rt2)
            idx += 1
        end
        push!(cones, CO.PosSemidef{T}(dim))
        push!(cone_idxs, 1:dim)
        cone_offset += dim

        for i in 1:(n - 1)
            L = size(PWts[i], 2)
            dim = div(L * (L + 1), 2)
            num_poly_vars += dim
            push!(a_poly, zeros(T, U, dim))
            idx = 1
            for k in 1:L, l in 1:k
                # off diagonals are doubled, but already scaled by rt2
                a_poly[i + 1][:, idx] = PWtsc[i][:, k] .* PWtsc[i][:, l] * (k == l ? 1 : rt2)
                idx += 1
            end
            push!(cone_idxs, cone_offset:(cone_offset + dim))
            push!(cones, CO.PosSemidef{T}(dim))
            cone_offset += dim
        end
        A_poly = hcat(a_poly...)
        A_poly = [-Matrix{T}(I, U, U) A_poly]
        c_poly = zeros(T, num_poly_vars)
        b_poly = zeros(T, U)
        G_poly = [zeros(T, num_poly_vars, U) -Matrix{T}(I, num_poly_vars, num_poly_vars)]
        h_poly = zeros(num_poly_vars)
    end

    if use_sumlog
        # pre-pad with one hypograph variable
        c_log = vcat(-1, zeros(T, U))
        dimx = 1 + U
        G_poly = [zeros(T, U, 1) G_poly]
        A = [
            zeros(T, size(A_poly, 1)) A_poly
            0 w'
            ]
        A = zeros(T, 1, dimx)
        A[1, 2:end] = w
        h_log = zeros(T, nobs + 2)
        h_log[2] = 1
        G_log = zeros(T, 2 + nobs, dimx)
        G_log[1, 1] = -1
        for i in 1:nobs
            G_log[i + 2, 2:end] = -basis_evals[i, :]
        end
        push!(cone_idxs, cone_offset:(cone_offset + 1 + nobs))
        push!(cones, CO.HypoPerSumLog{T}(nobs + 2, alpha = alpha))
    else
        # pre-pad with `nobs` hypograph variables
        c_log = vcat(-ones(T, nobs), zeros(T, U))
        dimx = nobs + U
        G_poly = [zeros(T, U, nobs) G_poly]
        A = [
            zeros(T, size(A_poly, 1), nobs) A_poly
            zeros(T, 1, nobs) w'
            ]
        h_log = zeros(T, 3 * nobs)

        G_log = zeros(T, 3 * nobs, dimx)
        offset = 1
        for i in 1:nobs
            G_log[offset, i] = -1.0
            G_log[offset + 2, (nobs + 1):end] = -basis_evals[i, :]
            h_log[offset + 1] = 1.0
            offset += 3
            push!(cones, CO.HypoPerSumLog{T}(3, alpha = -2))
            push!(cone_idxs, cone_offset:(cone_offset + 2))
            cone_offset += 3
        end
    end
    G = vcat(G_poly, G_log)
    h = vcat(h_poly, h_log)
    c = vcat(c_log, c_poly)
    b = vcat(b_poly, 1)

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

densityest(nobs::Int, n::Int, deg::Int; alpha = -1, options...) = densityest(randn(nobs, n), deg; alpha = alpha, options...)

# densityest1(; T::THR = Float64) = densityest(iris_data(), 4, use_sumlog = true, T = T)
# densityest2(; T::THR = Float64) = densityest(iris_data(), 4, use_sumlog = false, T = T)
# densityest3(; T::THR = Float64) = densityest(cancer_data(), 4, use_sumlog = true, T = T)
# densityest4(; T::THR = Float64) = densityest(cancer_data(), 4, use_sumlog = false, T = T)
densityest5(; T::THR = Float64) = densityest(200, 1, 4, use_sumlog = true, T = T)
densityest6(; T::THR = Float64) = densityest(200, 1, 4, use_sumlog = false, T = T)

function test_densityest(instance::Function; T::THR = Float64, rseed::Int = 1, options)
    Random.seed!(rseed)
    d = instance(T = T)
    model = MO.PreprocessedLinearModel{T}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver{T}(model; options...)
    SO.solve(solver)
    r = SO.get_certificates(solver, model, test = false, atol = 1e-4, rtol = 1e-4)
    # @test r.status == :Optimal
    return
end

test_densityest_all(; T::THR = Float64, options...) = test_densityest.([
    densityest1,
    densityest2,
    densityest3,
    densityest4,
    densityest5,
    densityest6,
    ], T = T, options = options)

test_densityest(; T::THR = Float64, options...) = test_densityest.([
    densityest5,
    densityest6,
    ], T = T, options = options)



# n_range = [1, 2]
# deg_range = [4, 6]
# tf = [true, false]
# seeds = 1:2
# real_types = [Float64, Float32]
# # real_types = [Float32]
# alpha_range = [2, 4, -1, -2]
#
# io = open("densityest.csv", "w")
# println(io, "usesumlog,real,alpha,seed,n,deg,dimx,dimy,dimz,time,bytes,numiters,status,pobj,dobj,xfeas,yfeas,zfeas")
# for n in n_range, deg in d_range, T in real_types, seed in seeds
#     for use_sumlog in tf
#         if use_sumlog
#             a_range = alpha_range
#         else
#             a_range = [-1]
#         end
#         for alpha in a_range
#             Random.seed!(seed)
#             d = densityest(200, n, deg, use_sumlog = use_sumlog, T = T, alpha = alpha)
#             model = MO.PreprocessedLinearModel{T}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
#             solver = SO.HSDSolver{T}(model, tol_abs_opt = 1e-5, tol_rel_opt = 1e-5, time_limit = 600)
#             t = @timed SO.solve(solver)
#             r = SO.get_certificates(solver, model, test = false, atol = 1e-4, rtol = 1e-4)
#             dimx = size(d.G, 2)
#             dimy = size(d.A, 1)
#             dimz = size(d.G, 1)
#             println(io, "$use_sumlog,$T,$alpha,$seed,$n,$deg,$dimx,$dimy,$dimz,$(t[2]),$(t[3])," *
#                 "$(solver.num_iters),$(r.status),$(r.primal_obj),$(r.dual_obj),$(solver.x_feas)," *
#                 "$(solver.y_feas),$(solver.z_feas)"
#                 )
#         end
#     end
# end
# close(io)



;
