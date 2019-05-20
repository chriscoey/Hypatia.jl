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

using LinearAlgebra
using SparseArrays
import Combinatorics
import Random
import Distributions
using Test
using Printf
using TimerOutputs
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
    sample_factor::Int = 10,
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
    @assert sample_factor >= 2
    num_samples = sample_factor * U
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
    sample_factor::Int = 10,
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
    @assert sample_factor >= 2
    num_samples = sample_factor * U
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

# TODO can just use P'*diag(x)*P in PSDCone constraints, where Ps come from data above
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
                PiPj = [conj(Pk[u, i]) * Pk[u, j] for u in 1:dat.U]
                if i == j
                    @. Gk[l, :] = -real(PiPj)
                else
                    @. Gk[l, :] = -rt2 * real(PiPj)
                    l += 1
                    @. Gk[l, :] = -rt2 * imag(PiPj)
                end
                l += 1
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
                PiPj = [Pk[u, i] * Pk[u, j] for u in 1:dat.U]
                if i == j
                    @. Gk[l, :] = -PiPj
                else
                    @. Gk[l, :] = -rt2 * PiPj
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

function solve_model(dat, is_min::Bool)
    model = MO.PreprocessedLinearModel(dat.c, dat.A, dat.b, dat.G, dat.h, dat.cones, dat.cone_idxs)
    solver = SO.HSDSolver(model, verbose = true, tol_feas = 1e-5, tol_rel_opt = 1e-6, tol_abs_opt = 1e-5, max_iters = 200)
    SO.solve(solver)

    obj = SO.get_primal_obj(solver)
    if !is_min
        obj = -obj
    end
    # @test SO.get_status(solver) == :Optimal

    return obj
    # TODO experimental minimum over random points in domain
    # F_min_pts = minimum(F_interp)
    # println("minimum on domain over interp points:\n", F_min_pts)
    # @test obj <= (primal_wsos ? -F_min_pts : F_min_pts)
end

function run_all(
    n::Int,
    d::Int,
    F_coef::Hermitian{ComplexF64, Matrix{ComplexF64}},
    F_fun::Function,
    inner_radius::Float64,
    outer_radius::Float64,
    )
    # setup real and complex interpolations
    # TODO time them, note the second uses more candidate points which may be unfair
    println()
    println("starting complex interpolation")
    C_interp = setup_C_interp(n, d, F_fun, inner_radius, outer_radius)
    println("C U is   ", C_interp.U)
    println("C Ls are ", size.(C_interp.P_data, 2))
    println("starting real interpolation")
    R_interp = setup_R_interp(n, d, F_fun, inner_radius, outer_radius)
    println("R U is   ", R_interp.U)
    println("R Ls are ", size.(R_interp.P_data, 2))
    println("finished interpolation")

    # build real and complex models
    println()
    println("building complex models")
    C_wsos_primal = build_wsos_primal(C_interp)
    C_wsos_dual = build_wsos_dual(C_interp)
    C_psd_dual = build_psd_dual(C_interp)
    println("building real models")
    R_wsos_primal = build_wsos_primal(R_interp)
    R_wsos_dual = build_wsos_dual(R_interp)
    R_psd_dual = build_psd_dual(R_interp)
    println("finished building models")

    # TODO also measure time (also the setup time for real or complex)
    # run models
    println()
    println("starting solves")
    objs = Dict()
    # println("C wsos primal:")
    # objs[:Cwp] = solve_model(C_wsos_primal, false)
    println("C wsos dual:")
    objs[:Cwd] = solve_model(C_wsos_dual, true)
    println("C psd dual:")
    objs[:Cp] = solve_model(C_psd_dual, true)
    # println("R wsos primal:")
    # objs[:Rwp] = solve_model(R_wsos_primal, false)
    println("R wsos dual:")
    objs[:Rwd] = solve_model(R_wsos_dual, true)
    println("R psd dual:")
    objs[:Rp] = solve_model(R_psd_dual, true)
    println("finished solves")

    # TODO check objs against minimum at test sample points

    return objs
end

function run_all_rand(
    n::Int,
    d::Int,
    deg::Int,
    inner_radius::Float64,
    outer_radius::Float64,
    )
    @assert deg <= d

    # generate random objective of degree deg
    (F_coef, F_fun) = rand_obj(n, deg)

    # run models of degree d
    objs = run_all(n, d, F_coef, F_fun, inner_radius, outer_radius)

    return objs
end

function speedtest(; rseed::Int = 1)
    Random.seed!(rseed)
    if !isdir("timings")
        mkdir("timings")
    end

    for nbhd in ["infty", "hess"]
        open(joinpath("timings", "polyannulus_" * nbhd * ".csv"), "a") do f
            @printf(f, "%15s, %15s, %15s, %15s, %15s, %15s, %15s, %15s, %15s, %15s, %15s, %15s, %15s, %15s, %15s, %15s\n",
            "poly", "obj", "n", "halfdeg", "G dim", "nu", "interp t", "build t", "solve t", "affine %t", "comb %t", "dir %t", "# iters", "# corr steps", "aff / iter",
            "comb / iter",
            )
        end
    end

    for n in [2, 3], halfdeg in [2]
        (F_coef, F_fun) = rand_obj(n, halfdeg - 1)
        for num_type in ["r", "c"]
            use_real = (num_type == "r")
            if use_real
                ti = @elapsed interp = setup_R_interp(n, halfdeg, F_fun, 0.5, 1.5)
            else
                ti = @elapsed interp = setup_C_interp(n, halfdeg, F_fun, 0.5, 1.5)
            end

            for cone_type in ["w", "p"]
                use_wsos = (cone_type == "w")
                if use_wsos
                    d = build_wsos_dual(interp)
                else
                    d = build_psd_dual(interp)
                end
                nu = sum(size(Pk, 2) for Pk in interp.P_data)

                for nbhd in ["infty", "hess"]
                    infty_nbhd = (nbhd == "infty")
                    build_time = 0
                    obj = 0
                    for _ in 1:2
                        reset_timer!(Hypatia.to)
                        build_time = @elapsed model = MO.PreprocessedLinearModel(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
                        stepper = SO.CombinedHSDStepper(model, infty_nbhd = infty_nbhd)
                        solver = SO.HSDSolver(model, stepper = stepper)
                        SO.solve(solver)
                        obj = SO.get_primal_obj(solver)
                    end
                    polyname = "$(cone_type)_$(num_type)_$(n)_$(halfdeg)"

                    open(joinpath("timings", polyname * nbhd * ".txt"), "w") do f
                        print_timer(f, Hypatia.to)
                    end

                    open(joinpath("timings", "polyannulus_" * nbhd * ".csv"), "a") do f
                        G1 = size(d.G, 1)
                        tt = TimerOutputs.tottime(Hypatia.to) # total solving time (nanoseconds)
                        tts = tt / 1e6
                        tb = build_time
                        ta = TimerOutputs.time(Hypatia.to["aff alpha"]) / tt # % of time in affine alpha
                        tc = TimerOutputs.time(Hypatia.to["comb alpha"]) / tt # % of time in comb alpha
                        td = TimerOutputs.time(Hypatia.to["directions"]) / tt # % of time calculating directions
                        num_iters = TimerOutputs.ncalls(Hypatia.to["directions"])
                        aff_per_iter = TimerOutputs.ncalls(Hypatia.to["aff alpha"]["linstep"]) / num_iters
                        comb_per_iter = TimerOutputs.ncalls(Hypatia.to["comb alpha"]["linstep"]) / num_iters

                        if "corr alpha" in keys(Hypatia.to.inner_timers)
                            num_corr = TimerOutputs.ncalls(Hypatia.to["corr alpha"])
                        else
                            num_corr = 0
                        end

                        @printf(f, "%15s, %15.3f, %15d, %15d, %15d, %15d, %15.2f, %15.2f, %15.2f, %15.2f, %15.2f, %15.2f, %15d, %15d, %15.2f, %15.2f\n",
                            polyname, obj, n, halfdeg, G1, nu, ti, tb, tts, ta, tc, td, num_iters, num_corr, aff_per_iter, comb_per_iter
                            )
                    end # do
                end # nbhd
            end # cone type
        end # numtype
    end # n halfdeg

    return
end
