#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using Test
import Random
using LinearAlgebra
import ForwardDiff
import Hypatia
const CO = Hypatia.Cones
const MU = Hypatia.ModelUtilities

function test_barrier_oracles(
    cone::CO.Cone{T},
    barrier::Function;
    noise::Real = 0.2,
    scale::Real = 10000,
    tol::Real = 100eps(T),
    init_tol::Real = tol,
    init_only::Bool = false,
    ) where {T <: Real}
    Random.seed!(1)
    @test !CO.use_scaling(cone)
    CO.setup_data(cone)
    dim = CO.dimension(cone)
    point = Vector{T}(undef, dim)

    if isfinite(init_tol)
        # tests for centrality of initial point
        CO.set_initial_point(point, cone)
        CO.load_point(cone, point)
        @test cone.point == point
        @test CO.is_feas(cone)
        grad = CO.grad(cone)
        @test dot(point, -grad) ≈ norm(point) * norm(grad) atol=init_tol rtol=init_tol
        @test point ≈ -grad atol=init_tol rtol=init_tol
    end
    init_only && return

    # tests for perturbed point
    CO.reset_data(cone)
    CO.set_initial_point(point, cone)
    if !iszero(noise)
        point += T(noise) * (rand(T, dim) .- inv(T(2)))
    end
    point /= scale

    CO.load_point(cone, point)
    @test cone.point == point
    @test CO.is_feas(cone)

    grad = CO.grad(cone)
    nu = CO.get_nu(cone)
    @test dot(point, grad) ≈ -nu atol=tol rtol=tol
    hess = CO.hess(cone)
    @test hess * point ≈ -grad atol=tol rtol=tol

    inv_hess = CO.inv_hess(cone)
    @test hess * inv_hess ≈ I atol=tol rtol=tol

    prod = similar(point)
    @test CO.hess_prod!(prod, point, cone) ≈ -grad atol=tol rtol=tol
    @test CO.inv_hess_prod!(prod, grad, cone) ≈ -point atol=tol rtol=tol
    prod_mat = similar(point, dim, dim)
    @test CO.hess_prod!(prod_mat, Matrix(inv_hess), cone) ≈ I atol=tol rtol=tol
    @test CO.inv_hess_prod!(prod_mat, Matrix(hess), cone) ≈ I atol=tol rtol=tol

    # compare to ForwardDiff
    # check gradient and Hessian agree
    @test ForwardDiff.gradient(barrier, point) ≈ grad atol=tol rtol=tol
    @test ForwardDiff.hessian(barrier, point) ≈ hess atol=tol rtol=tol
    if CO.use_3order_corr(cone)
        # check 3rd order corrector agrees
        FD_3deriv = ForwardDiff.jacobian(x -> ForwardDiff.hessian(barrier, x), point)
        # check log-homog property that F'''(point)[point] = -2F''(point)
        @test reshape(FD_3deriv * point, dim, dim) ≈ -2 * hess
        # check correction term agrees with directional 3rd derivative
        (primal_dir, dual_dir) = (randn(dim), randn(dim))
        Hinv_dual_dir = CO.inv_hess_prod!(similar(dual_dir), dual_dir, cone)
        FD_corr = reshape(FD_3deriv * primal_dir, dim, dim) * Hinv_dual_dir / -2
        @test FD_corr ≈ CO.correction(cone, primal_dir, dual_dir) atol=tol rtol=tol
    end

    dual_point = -grad
    if !iszero(noise)
        dual_point += T(noise) * (rand(T, dim) .- inv(T(2)))
    end
    # CO.load_dual_point(cone, dual_point)
    # @test cone.dual_point == dual_point
    # scal_hess = CO.scal_hess(cone, one(T))
    # @show scal_hess
    # @show eigvals(scal_hess)
    # TODO add tests for scal hess correctness

    return
end

# TODO cleanup, cover all scaling oracles, comment in some places
function test_barrier_scaling_oracles(
    cone::CO.Cone{T},
    barrier::Function;
    noise::Real = 0.2,
    tol::Real = 100eps(T),
    ) where {T <: Real}
    Random.seed!(1)
    @test CO.use_scaling(cone)
    CO.setup_data(cone)
    CO.reset_data(cone)
    dim = CO.dimension(cone)

    point = Vector{T}(undef, dim)
    dual_point = Vector{T}(undef, dim)
    CO.set_initial_point(point, cone)
    CO.set_initial_point(dual_point, cone)
    if !iszero(noise)
        point += T(noise) * (rand(T, dim) .- inv(T(2)))
        dual_point += T(noise) * (rand(T, dim) .- inv(T(2)))
    end

    CO.load_point(cone, point)
    CO.load_dual_point(cone, dual_point)
    @test cone.point == point
    @test cone.dual_point == dual_point
    @test CO.is_feas(cone)

    grad = CO.grad(cone)
    # hess and inv_hess oracles, not the same as for non-scaling tests
    hess = CO.hess(cone)
    @test hess * cone.point ≈ cone.dual_point atol=tol rtol=tol
    inv_hess = CO.inv_hess(cone)
    @test inv_hess * cone.dual_point ≈ cone.point atol=tol rtol=tol
    @test hess * inv_hess ≈ I atol=tol rtol=tol
    # hess and inv_hes product oracles
    prod_mat = similar(point, dim, dim)
    @test CO.hess_prod!(prod_mat, Matrix(inv_hess), cone) ≈ I atol=tol rtol=tol
    @test CO.inv_hess_prod!(prod_mat, Matrix(hess), cone) ≈ I atol=tol rtol=tol

    # multiplication and division by scaling matrix W
    λ = CO.scalmat_prod!(similar(cone.point), cone.dual_point, cone)
    W = CO.scalmat_prod!(similar(point, dim, dim), Matrix{T}(I, dim, dim), cone)
    @test W' * λ ≈ cone.point atol=tol rtol=tol
    prod = similar(point)
    @test CO.scalmat_ldiv!(prod, cone.point, cone, trans = true) ≈ λ atol=tol rtol=tol
    @test CO.scalmat_ldiv!(prod_mat, W, cone, trans = false) ≈ I atol=tol rtol=tol

    # additional sanity checks
    @test W' * W ≈ inv_hess atol=tol rtol=tol
    WWz = CO.inv_hess_prod!(prod, cone.dual_point, cone)
    Wλ = CO.scalmat_prod!(prod, λ, cone)
    @test WWz ≈ Wλ atol=tol rtol=tol

    # conic product oracle and conic division by the scaled point λ
    e1 = CO.set_initial_point(zeros(T, dim), cone)
    λinv = CO.scalvec_ldiv!(similar(e1), e1, cone)
    @test CO.conic_prod!(similar(e1), λinv, λ, cone) ≈ e1 atol=tol rtol=tol

    # e1 = λ \circ W * -grad tested in different ways
    @test e1 ≈ CO.conic_prod!(prod, λ, -W * grad, cone) atol=tol rtol=tol
    @test -grad ≈ W \ CO.scalvec_ldiv!(prod, e1, cone) atol=tol rtol=tol
    @test -grad ≈ CO.scalmat_ldiv!(similar(e1), CO.scalvec_ldiv!(prod, e1, cone), cone, trans = false) atol=tol rtol=tol

    # max step in a recession direction
    max_step = CO.step_max_dist(cone, e1, e1)
    @test max_step ≈ T(Inf) atol=tol rtol=tol

    # max step elsewhere
    primal_dir = -e1 + T(noise) * (rand(T, dim) .- inv(T(2)))
    dual_dir = -e1 + T(noise) * (rand(T, dim) .- inv(T(2)))
    prev_primal = copy(cone.point)
    prev_dual = copy(cone.dual_point)

    # λ \circ W * correction = actual Mehrotra term
    mehrotra_term = CO.conic_prod!(similar(prod), W' \ primal_dir, W * dual_dir, cone)
    correction = CO.correction(cone, primal_dir, dual_dir)
    @test correction ≈ W \ CO.scalvec_ldiv!(similar(prod), mehrotra_term, cone)
    @test CO.conic_prod!(similar(e1), λ, W * correction, cone) ≈ mehrotra_term atol=tol rtol=tol

    # max step tests for these new directions
    max_step = CO.step_max_dist(cone, primal_dir, dual_dir)
    # check smaller step returns feasible iterates
    primal_feas = load_reset_and_check(cone, prev_primal + T(0.99) * max_step * primal_dir)
    dual_feas = load_reset_and_check(cone, prev_dual + T(0.99) * max_step * dual_dir)
    @test primal_feas && dual_feas
    # check larger step returns infeasible iterates
    primal_feas = load_reset_and_check(cone, prev_primal + T(1.01) * max_step * primal_dir)
    dual_feas = load_reset_and_check(cone, prev_dual + T(1.01) * max_step * dual_dir)
    @test !primal_feas || !dual_feas

    # identities from page 7 Myklebust and Tuncel, Interior Point Algorithms for Convex Optimization based on Primal-Dual Metrics
    check_feas(x) = (isfinite(barrier(x)) ? true : false)
    @test load_reset_and_check(cone, prev_primal)
    grad = CO.grad(cone)
    @test -CO.conjugate_gradient(barrier, check_feas, -grad) ≈ cone.point atol=tol rtol=tol
    conj_grad = CO.conjugate_gradient(barrier, check_feas, cone.dual_point)
    @test load_reset_and_check(cone, -conj_grad)
    grad = CO.grad(cone)
    @test -grad ≈ cone.dual_point atol=cbrt(eps(T)) rtol=cbrt(eps(T))

    return
end

function load_reset_and_check(cone, point)
    CO.load_point(cone, point)
    CO.reset_data(cone)
    return CO.is_feas(cone)
end

function test_nonnegative_barrier(T::Type{<:Real})
    barrier = (s -> -sum(log, s))
    for dim in [1, 3, 6]
        test_barrier_oracles(CO.Nonnegative{T}(dim, use_scaling = false), barrier)
        test_barrier_scaling_oracles(CO.Nonnegative{T}(dim, use_scaling = true), barrier)
    end
    return
end

function test_epinorminf_barrier(T::Type{<:Real})
    function barrier(s)
        (u, w) = (s[1], s[2:end])
        return -sum(log(u - abs2(wj) / u) for wj in w) - log(u)
    end
    for dim in [2, 4]
        test_barrier_oracles(CO.EpiNormInf{T}(dim), barrier)
    end
    return
end

function test_epinormeucl_barrier(T::Type{<:Real})
    function barrier(s)
        (u, w) = (s[1], s[2:end])
        return -log(abs2(u) - sum(abs2, w)) / 2
    end
    for dim in [2, 4, 6]
        test_barrier_oracles(CO.EpiNormEucl{T}(dim, use_scaling = false), barrier)
        test_barrier_scaling_oracles(CO.EpiNormEucl{T}(dim, use_scaling = true), barrier)
    end
    return
end

function test_epipersquare_barrier(T::Type{<:Real})
    function barrier(s)
        (u, v, w) = (s[1], s[2], s[3:end])
        return -log(2 * u * v - sum(abs2, w))
    end
    for dim in [3, 5]
        test_barrier_oracles(CO.EpiPerSquare{T}(dim), barrier)
    end
    return
end

function test_hypoperlog_barrier(T::Type{<:Real})
    function barrier(s)
        (u, v, w) = (s[1], s[2], s[3:end])
        return -log(v * sum(log(wj / v) for wj in w) - u) - sum(log, w) - log(v)
    end
    for dim in [3, 5, 10]
        test_barrier_oracles(CO.HypoPerLog{T}(dim), barrier, init_tol = 1e-5)
    end
    for dim in [15, 65, 75, 100, 500]
        test_barrier_oracles(CO.HypoPerLog{T}(dim), barrier, init_tol = 1e-1, init_only = true)
    end
    return
end

function test_epiperexp3_barrier(T::Type{<:Real})
    function barrier(s)
        (u, v, w) = (s[1], s[2], s[3])
        return -log(v * log(u / v) - w) - log(u) - log(v)
    end
    test_barrier_oracles(CO.EpiPerExp3{T}(), barrier, init_tol = 1e-6)
    # CO.EpiPerExp3{T}(false, use_scaling = true)
    # test_barrier_scaling_oracles(CO.EpiPerExp3{T}(false, use_scaling = true), barrier)
    return
end

function test_epiperexp_barrier(T::Type{<:Real})
    function barrier(s)
        (u, v, w) = (s[1], s[2], s[3:end])
        return -log(v * log(u / v) - v * log(sum(wi -> exp(wi / v), w))) - log(u) - log(v)
    end
    for dim in [3, 5, 10]
        test_barrier_oracles(CO.EpiPerExp{T}(dim), barrier, init_tol = 1e-5)
    end
    # NOTE when initial point improved, take tests up to dim=500 and tighten tolerance
    for dim in [15, 35 , 45, 100, 120, 200]
        test_barrier_oracles(CO.EpiPerExp{T}(dim), barrier, init_tol = 7e-1, init_only = true)
    end
    return
end

function test_power_barrier(T::Type{<:Real})
    Random.seed!(1)
    for (m, n) in [(2, 1), (2, 3), (4, 1), (4, 4)]
        alpha = rand(T, m) .+ 1
        alpha ./= sum(alpha)
        function barrier(s)
            (u, w) = (s[1:m], s[(m + 1):end])
            return -log(prod(u[j] ^ (2 * alpha[j]) for j in eachindex(u)) - sum(abs2, w)) - sum((1 - alpha[j]) * log(u[j]) for j in eachindex(u))
        end
        test_barrier_oracles(CO.Power{T}(alpha, n), barrier)
    end
    return
end

function test_hypogeomean_barrier(T::Type{<:Real})
    Random.seed!(1)
    for dim in [2, 3, 5, 15, 90, 120, 500]
        alpha = rand(T, dim - 1) .+ 1
        alpha ./= sum(alpha)
        function barrier(s)
            (u, w) = (s[1], s[2:end])
            return -log(prod((w[j] / alpha[j]) ^ alpha[j] for j in eachindex(w)) + u) - sum((1 - alpha[j]) * log(w[j] / alpha[j]) for j in eachindex(w)) - log(-u)
        end
        cone = CO.HypoGeomean{T}(alpha)
        if dim <= 3
            test_barrier_oracles(cone, barrier, init_tol = 1e-2)
        else
            test_barrier_oracles(cone, barrier, init_tol = 3e-1, init_only = true)
        end
    end
    return
end

function test_epinormspectral_barrier(T::Type{<:Real})
    for (n, m) in [(1, 2), (2, 2), (2, 3), (3, 5)]
        function barrier(s)
            (u, W) = (s[1], reshape(s[2:end], n, m))
            return -logdet(cholesky!(Symmetric(u * I - W * W' / u))) - log(u)
        end
        test_barrier_oracles(CO.EpiNormSpectral{T}(n, m), barrier)
    end
    return
end

function test_possemideftri_barrier(T::Type{<:Real})
    for side in [1, 2, 5]
        # real PSD cone
        function R_barrier(s)
            S = similar(s, side, side)
            CO.svec_to_smat!(S, s, sqrt(T(2)))
            return -logdet(cholesky!(Symmetric(S, :U)))
        end
        dim = div(side * (side + 1), 2)
        test_barrier_oracles(CO.PosSemidefTri{T, T}(dim, use_scaling = false), R_barrier)
        test_barrier_scaling_oracles(CO.PosSemidefTri{T, T}(dim, use_scaling = true), R_barrier)
        # complex PSD cone
        function C_barrier(s)
            S = zeros(Complex{eltype(s)}, side, side)
            CO.svec_to_smat!(S, s, sqrt(T(2)))
            return -logdet(cholesky!(Hermitian(S, :U)))
        end
        dim = side^2
        test_barrier_oracles(CO.PosSemidefTri{T, Complex{T}}(dim, use_scaling = false), C_barrier)
        test_barrier_scaling_oracles(CO.PosSemidefTri{T, Complex{T}}(dim, use_scaling = true), C_barrier)
    end
    return
end

function test_hypoperlogdettri_barrier(T::Type{<:Real})
    for side in [1, 2, 3, 4, 5, 6, 12, 20]
        function barrier(s)
            (u, v, W) = (s[1], s[2], similar(s, side, side))
            CO.vec_to_mat_U!(W, s[3:end])
            return -log(v * logdet(cholesky!(Symmetric(W / v, :U))) - u) - logdet(cholesky!(Symmetric(W, :U))) - log(v)
        end
        dim = 2 + div(side * (side + 1), 2)
        cone = CO.HypoPerLogdetTri{T}(dim)
        if side <= 5
            test_barrier_oracles(cone, barrier, init_tol = 1e-5)
        else
            test_barrier_oracles(cone, barrier, init_tol = 1e-1, init_only = true)
        end
    end
    return
end

function test_wsospolyinterp_barrier(T::Type{<:Real})
    Random.seed!(1)
    for (n, halfdeg) in [(1, 1), (1, 2), (1, 3), (2, 2), (3, 2), (2, 3)]
        # TODO test with more Pi matrices
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
        P0 = convert(Matrix{T}, P0)
        function barrier(s)
            Lambda = Symmetric(P0' * Diagonal(s) * P0)
            return -logdet(cholesky!(Lambda))
        end
        cone = CO.WSOSPolyInterp{T, T}(U, [P0], true)
        test_barrier_oracles(cone, barrier, init_tol = Inf) # TODO center and test initial points
    end
    # TODO also test complex case CO.WSOSPolyInterp{T, Complex{T}} - need complex MU interp functions first
    return
end

# function test_wsospolyinterpmat_barrier(T::Type{<:Real})
#     Random.seed!(1)
#     for n in 1:3, halfdeg in 1:3, R in 1:3
#         (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
#         P0 = convert(Matrix{T}, P0)
#         cone = CO.WSOSPolyInterpMat{T}(R, U, [P0], true)
#         test_barrier_oracles(cone)
#     end
#     return
# end
#
# function test_wsospolyinterpsoc_barrier(T::Type{<:Real})
#     Random.seed!(1)
#     for n in 1:2, halfdeg in 1:2, R in 3:3
#         (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
#         P0 = convert(Matrix{T}, P0)
#         cone = CO.WSOSPolyInterpSOC{T}(R, U, [P0], true)
#         test_barrier_oracles(cone)
#     end
#     return
# end
