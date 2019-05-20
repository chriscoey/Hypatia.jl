#=
Copyright 2019, Chris Coey and contributors

minimizes a real-valued complex polynomial over a domain defined by real-valued complex polynomials

TODO high priority
- write SDP formulations (for complex, maybe want a new complex hermitian PSD cone)
- try chebyshev-like V columns to improve numerics for high degree
- verify objective values lower bound values at sample points from domains
- compare two-sided vs squared domain formulations
- time the setup and solve

TODO low priority
- try getting points from box and then transforming only for the g_i
- different domains
- visualizations of domains and points
- write complex polys with known opt solutions for testing
- in dual formulation allow 1.0I as structured matrix
=#

import Pkg
Pkg.activate(".") # run from hypatia main directory level

using LinearAlgebra
using SparseArrays
import Combinatorics
import Random
import Distributions
using Test
using Printf
using TimerOutputs
const TO = TimerOutputs
import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MO = HYP.Models
const SO = HYP.Solvers

const rt2 = sqrt(2)

Random.seed!(1)

mon_pow(z, ex) = prod(z[i]^ex[i] for i in eachindex(ex))

# generate random real-valued complex objective function (as Hermitian matrix) of degree deg
# and provide function for evaluation
function rand_obj(n::Int, deg::Int)
    L = binomial(n + deg, n)
    F_coef = Hermitian(randn(ComplexF64, L, L))

    L_basis = [a for t in 0:deg for a in Combinatorics.multiexponents(n, t)]
    @assert length(L_basis) == L
    F_fun(z) = real(sum(F_coef[k, l] * mon_pow(z, L_basis[k]) * mon_pow(conj(z), L_basis[l]) for l in eachindex(L_basis) for k in eachindex(L_basis)))

    return (F_coef, F_fun)
end

function sample_in_annulus(num_samples::Int, n::Int, inner_radius::Float64, outer_radius::Float64)
    sample_at_radius(r::Float64) = (a = rand(); complex(r * cospi(a), r * sinpi(a)))

    # select some points from boundary
    num_boundary = div(num_samples, 10) + 1
    pts_inner_boundary = [[sample_at_radius(inner_radius) for i in 1:n] for u in 1:num_boundary]
    pts_outer_boundary = [[sample_at_radius(outer_radius) for i in 1:n] for u in 1:num_boundary]

    # select some points from interior
    num_interior = num_samples - 2 * num_boundary
    distr = Distributions.Uniform(inner_radius^2, outer_radius^2)
    pts_interior = [[sample_at_radius(sqrt(rand(distr))) for i in 1:n] for u in 1:num_interior]

    # sanity check
    pts = vcat(pts_inner_boundary, pts_outer_boundary, pts_interior)
    @assert all(r -> all(inner_radius - 1e-5 .<= abs.(r) .<= outer_radius + 1e-5), pts)

    return pts
end

# select subset of points to maximize |det(V)| in heuristic QR-based procedure
function select_subset_pts(U, V_basis, cand_pts)
    cand_V = [b(z) for z in cand_pts, b in V_basis]
    F = qr(Matrix(transpose(cand_V)), Val(true))
    # @test rank(F) == U
    keep = F.p[1:U]
    pts = cand_pts[keep]
    V = cand_V[keep, :]
    return (pts, V)
end

# TODO allow different radius for each dimension
function setup_C_interp(
    n::Int,
    d::Int,
    F_fun::Function,
    inner_radius::Float64,
    outer_radius::Float64;
    # sample_factor::Int = 10,
    use_QR::Bool = true,
    )
    # generate interpolation
    L = binomial(n + d, n)
    U = L^2
    L_basis = [a for t in 0:d for a in Combinatorics.multiexponents(n, t)]
    @assert length(L_basis) == L
    V_basis = [z -> mon_pow(z, L_basis[k]) * mon_pow(conj(z), L_basis[l]) for l in eachindex(L_basis) for k in eachindex(L_basis)]
    @assert length(V_basis) == U

    # select some points from annulus domain
    # @assert sample_factor >= 2
    # num_samples = sample_factor * U
    num_samples = 6000
    @show U
    cand_pts = sample_in_annulus(num_samples, n, inner_radius, outer_radius)
    (pts, V) = select_subset_pts(U, V_basis, cand_pts)

    # setup objective data
    F_interp = F_fun.(pts)
    @show minimum(F_interp)

    # setup WSOS constraint data
    # TODO try instead getting points from box and then transforming only for the g_i
    # TODO compare single squared constraint vs two bound constraints (changes size of P_g too)
    P0 = V[:, 1:L]
    if use_QR
        P0 = Matrix(qr(P0).Q)
    end
    P_data = [P0]
    P0g = V[:, 1:binomial(n + d - 1, n)]
    for i in 1:n
        g_inner = map(z -> sqrt(max(0.0, abs2(z[i]) - inner_radius^2)), pts)
        g_outer = map(z -> sqrt(max(0.0, outer_radius^2 - abs2(z[i]))), pts)
        for gi in [g_inner, g_outer]
            Pi = Diagonal(gi) * P0g
            if use_QR
                Pi = Matrix(qr(Pi).Q)
            end
            push!(P_data, Pi)
        end
    end

    return (U=U, F_interp=F_interp, P_data=P_data)
end

function setup_R_interp(
    n::Int,
    d::Int,
    F_fun::Function,
    inner_radius::Float64,
    outer_radius::Float64;
    # sample_factor::Int = 10,
    use_QR::Bool = true,
    )
    # generate interpolation
    # TODO use more numerically stable column basis
    L = binomial(2n + d, 2n)
    U = binomial(2n + 2d, 2n)
    L_basis = [a for t in 0:d for a in Combinatorics.multiexponents(2n, t)]
    @assert length(L_basis) == L
    V_basis = [z -> mon_pow(z, a) for t in 0:2d for a in Combinatorics.multiexponents(2n, t)]
    @assert length(V_basis) == U

    # select some points from annulus domain
    # @assert sample_factor >= 2
    # num_samples = sample_factor * U
    num_samples = 6000
    cand_pts_complex = sample_in_annulus(num_samples, n, inner_radius, outer_radius)
    cand_pts = [vcat(real(z), imag(z)) for z in cand_pts_complex]
    (pts, V) = select_subset_pts(U, V_basis, cand_pts)

    # setup objective data
    F_interp = [F_fun(complex.(p[1:n], p[n+1:2n])) for p in pts]
    @show minimum(F_interp)

    # setup WSOS constraint data
    # TODO try instead getting points from box and then transforming only for the g_i
    # TODO compare single squared constraint vs two bound constraints (changes size of P_g too)
    P0 = V[:, 1:L]
    if use_QR
        P0 = Matrix(qr(P0).Q)
    end
    P_data = [P0]
    P0g = V[:, 1:binomial(2n + d - 1, 2n)]
    for i in 1:n
        g_inner = map(p -> sqrt(max(0.0, p[i]^2 + p[n+i]^2 - inner_radius^2)), pts)
        g_outer = map(p -> sqrt(max(0.0, outer_radius^2 - p[i]^2 - p[n+i]^2)), pts)
        for gi in [g_inner, g_outer]
            Pi = Diagonal(gi) * P0g
            if use_QR
                Pi = Matrix(qr(Pi).Q)
            end
            push!(P_data, Pi)
        end
    end

    return (U=U, F_interp=F_interp, P_data=P_data)
end

function build_wsos_primal(dat)
    c = [-1.0]
    A = zeros(0, 1)
    b = Float64[]
    G = ones(dat.U, 1)
    h = dat.F_interp
    cones = [CO.WSOSPolyInterp(dat.U, dat.P_data, false)]
    cone_idxs = [1:dat.U]
    return (c=c, A=A, b=b, G=G, h=h, cones=cones, cone_idxs=cone_idxs)
end

function build_wsos_dual(dat; elim::Bool = false)
    if elim
        # eliminate the equality and the first variable
        # x₁ = 1 - x₂ - ... - xᵤ
        # TODO handle constant term in obj
        @show dat.F_interp[1] # constant
        n = dat.U - 1
        c = dat.F_interp[2:end] .- dat.F_interp[1]
        A = zeros(0, n)
        b = Float64[]
        GI = vcat(fill(1, n), collect(2:dat.U))
        GJ = vcat(collect(1:n), collect(1:n))
        GV = vcat(fill(1.0, n), fill(-1.0, n))
        G = sparse(GI, GJ, GV, dat.U, n)
        h = zeros(dat.U); h[1] = 1.0
    else
        c = dat.F_interp
        A = ones(1, dat.U)
        b = [1.0]
        G = Diagonal(-1.0I, dat.U) # TODO use -1.0I
        h = zeros(dat.U)
    end
    cones = [CO.WSOSPolyInterp(dat.U, dat.P_data, true)]
    cone_idxs = [1:dat.U]
    return (c=c, A=A, b=b, G=G, h=h, cones=cones, cone_idxs=cone_idxs)
end

function build_psd_dual(dat)
    c = dat.F_interp
    A = ones(1, dat.U)
    b = [1.0]
    Gs = Matrix{Float64}[]
    cones = CO.Cone[]
    cone_idxs = UnitRange{Int}[]
    if eltype(dat.P_data[1]) <: Complex
        # complex interp
        # TODO option for real or complex PSD cone
        rowidx = 1
        for Pk in dat.P_data
            Lk = size(Pk, 2)
            dk = Lk^2
            push!(cone_idxs, rowidx:(rowidx + dk - 1))
            push!(cones, CO.PosSemidef{ComplexF64}(dk))
            Gk = Matrix{Float64}(undef, dk, dat.U)
            l = 1
            for i in 1:Lk, j in 1:i
                if i == j
                    for u in 1:dat.U
                        @inbounds Gk[l, u] = -real(conj(Pk[u, i]) * Pk[u, j])
                    end
                    l += 1
                else
                    for u in 1:dat.U
                        Pkuij = -rt2 * conj(Pk[u, i]) * Pk[u, j]
                        @inbounds Gk[l, u] = real(Pkuij)
                        @inbounds Gk[l + 1, u] = imag(Pkuij)
                    end
                    l += 2
                end
            end
            push!(Gs, Gk)
            rowidx += dk
        end
    else
        # real interp
        rowidx = 1
        for Pk in dat.P_data
            Lk = size(Pk, 2)
            dk = div(Lk * (Lk + 1), 2)
            push!(cone_idxs, rowidx:(rowidx + dk - 1))
            push!(cones, CO.PosSemidef{Float64}(dk))
            Gk = Matrix{Float64}(undef, dk, dat.U)
            l = 1
            for i in 1:Lk, j in 1:i
                scal = (i == j) ? -1.0 : -rt2
                for u in 1:dat.U
                    @inbounds Gk[l, u] = scal * Pk[u, i] * Pk[u, j]
                end
                l += 1
            end
            push!(Gs, Gk)
            rowidx += dk
        end
    end
    G = vcat(Gs...)
    h = zeros(size(G, 1))
    return (c=c, A=A, b=b, G=G, h=h, cones=cones, cone_idxs=cone_idxs)
end

function speedtest(n::Int, halfdeg::Int, maxU::Int; rseed::Int = 1)
    (F_coef, F_fun) = rand_obj(n, halfdeg - 1)
    for is_complex in [true, false]
        str_is_complex = is_complex ? "c" : "r"
        U = is_complex ? binomial(2n + 2halfdeg, 2n) : binomial(n + 2halfdeg, n)
        if U > maxU
            println("skipping n=$n, halfdeg=$halfdeg, since U=$U")
            continue
        end
        interp_fun = is_complex ? setup_C_interp : setup_R_interp
        ti = @elapsed interp = interp_fun(n, halfdeg, F_fun, 0.8, 1.2)

        for is_wsos in [true, false]
            str_is_wsos = is_wsos ? "w" : "p"
            model_fun = is_wsos ? build_wsos_dual : build_psd_dual
            d = model_fun(interp)
            barpar = sum(size(Pk, 2) for Pk in interp.P_data)
            (dim_cone, num_vars) = size(d.G)

            for is_infty_nbhd in [true]
                str_is_infty_nbhd = is_infty_nbhd ? "i" : "h"
                modelname = "$(str_is_complex)_$(str_is_wsos)_$(str_is_infty_nbhd)_$(n)_$(halfdeg)"

                reset_timer!(Hypatia.to)
                println("building Hypatia internal model")
                build_time = @elapsed model = MO.PreprocessedLinearModel(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
                stepper = SO.CombinedHSDStepper(model, infty_nbhd = is_infty_nbhd)
                solver = SO.HSDSolver(model, stepper = stepper,
                    tol_rel_opt = 1e-5, tol_abs_opt = 1e-5, tol_feas = 1e-5,
                    time_limit = 1800.0, max_iters = 250,)
                SO.solve(solver)
                pobj = SO.get_primal_obj(solver)
                dobj = SO.get_dual_obj(solver)
                status = SO.get_status(solver)

                open(joinpath("timings", modelname * ".txt"), "w") do f
                    print_timer(f, Hypatia.to)
                end

                open(joinpath("timings", "results.csv"), "a") do f
                    tt = TO.tottime(Hypatia.to) # total solving time (nanoseconds)
                    tts = tt / 1e9
                    tb = build_time
                    ta = TO.time(Hypatia.to["aff alpha"]) / tt # % of time in affine alpha
                    tc = TO.time(Hypatia.to["comb alpha"]) / tt # % of time in comb alpha
                    td = TO.time(Hypatia.to["directions"]) / tt # % of time calculating directions
                    num_iters = TO.ncalls(Hypatia.to["directions"])
                    aff_per_iter = TO.ncalls(Hypatia.to["aff alpha"]["linstep"]) / num_iters
                    comb_per_iter = TO.ncalls(Hypatia.to["comb alpha"]["linstep"]) / num_iters
                    num_corr = 0
                    if haskey(Hypatia.to.inner_timers, "corr alpha")
                        num_corr = TO.ncalls(Hypatia.to["corr alpha"])
                    end

                    @printf(f, "%s,%s,%s,%d,%d,%d,%d,%d,%s,%f,%f,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f\n",
                        str_is_complex, str_is_wsos, str_is_infty_nbhd,
                        n, halfdeg, dim_cone, num_vars, barpar,
                        status, pobj, dobj,
                        num_iters, num_corr, aff_per_iter, comb_per_iter,
                        ti, tb, tts, ta, tc, td
                        )
                end # do
            end # nbhd
        end # cone type
    end # numtype

    return
end


if !isdir("timings")
    mkdir("timings")
end

# compile run
println("\ncompile run\n")
speedtest(2, 2, 100)
println("\n")

open(joinpath("timings", "results.csv"), "w") do f
    @printf(f, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
    "RorC", "PorW", "HorI",
    "n", "half_deg", "dim_cone", "num_vars", "bar_par",
    "status", "p_obj", "d_obj",
    "iters", "corr_steps", "avg_aff_ls", "avg_comb_ls",
    "t_interp", "t_build", "t_solve", "frac_t_aff", "frac_t_comb", "frac_t_dir",
    )
end

# full run
ns = [1,2,3,4,6,8,10]
halfdegs = [1,2,3,4,6,8,10,15,20]
# ns = [1,2,3,4]
# halfdegs = [1,2,3,4,6,8]
maxU = 5000

@show ns
@show halfdegs
@show maxU

for n in ns, halfdeg in halfdegs
    println("\n")
    @show n, halfdeg
    println()
    try
        speedtest(n, halfdeg, maxU)
    catch e
        println(e)
    end
    println()
end
