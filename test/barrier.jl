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
        point /= scale
    end

    CO.load_point(cone, point)
    @test cone.point == point
    @test CO.is_feas(cone)

    # CO.update_grad(cone)
    grad = CO.grad(cone)
    nu = CO.get_nu(cone)
    @test dot(point, grad) ≈ -nu atol=tol rtol=tol
    hess = CO.hess(cone)
    @test hess * point ≈ -grad atol=tol rtol=tol

    if T in (Float32, Float64) # NOTE can only use BLAS floats with ForwardDiff barriers
        @test ForwardDiff.gradient(barrier, point) ≈ grad atol=tol rtol=tol
        @test ForwardDiff.hessian(barrier, point) ≈ hess atol=tol rtol=tol
    end

    inv_hess = CO.inv_hess(cone)
    @test hess * inv_hess ≈ I atol=tol rtol=tol

    # CO.update_hess_prod(cone)
    # CO.update_inv_hess_prod(cone)
    prod = similar(point)
    @test CO.hess_prod!(prod, point, cone) ≈ -grad atol=tol rtol=tol
    @test CO.inv_hess_prod!(prod, grad, cone) ≈ -point atol=tol rtol=tol
    prod_mat = similar(point, dim, dim)
    @test CO.hess_prod!(prod_mat, Matrix(inv_hess), cone) ≈ I atol=tol rtol=tol
    @test CO.inv_hess_prod!(prod_mat, Matrix(hess), cone) ≈ I atol=tol rtol=tol

    return
end

# TODO cleanup, cover all scaling oracles, comment in some places
function test_barrier_scaling_oracles(
    cone::CO.Cone{T};
    noise::Real = 0.2,
    scale::Real = 10000,
    tol::Real = 100eps(T),
    ) where {T <: Real}
    @test CO.use_scaling(cone)
    CO.setup_data(cone)
    CO.reset_data(cone)
    dim = CO.dimension(cone)
    point = Vector{T}(undef, dim)
    dual_point = Vector{T}(undef, dim)

    Random.seed!(1)

    CO.set_initial_point(point, cone)
    CO.set_initial_point(dual_point, cone)
    point .+= T(noise) * (rand(T, dim) .- inv(T(2))) / scale
    dual_point .+= T(noise) * (rand(T, dim) .- inv(T(2))) / scale
    @show point, dual_point

    CO.load_point(cone, point)
    # TODO why isn't load_dual_point implemented?
    cone.dual_point .= dual_point
    @test cone.point == point
    @test cone.dual_point == dual_point
    @test CO.is_feas(cone)

    grad = CO.grad(cone)
    cone.use_scaling = true # TODO update when it's an option, run these tests optionally
    # not the same hessian and inverse hessians as for non-scaling tests
    hess = CO.hess(cone)
    @test hess * cone.point ≈ cone.dual_point atol=tol rtol=tol
    inv_hess = CO.inv_hess(cone)
    @test inv_hess * cone.dual_point ≈ cone.point atol=tol rtol=tol
    @test hess * inv_hess ≈ I atol=tol rtol=tol
    prod_mat = similar(point, dim, dim)
    @test CO.hess_prod!(prod_mat, Matrix(inv_hess), cone) ≈ I atol=tol rtol=tol
    @test CO.inv_hess_prod!(prod_mat, Matrix(hess), cone) ≈ I atol=tol rtol=tol

    λ = similar(cone.point)
    CO.scalmat_prod!(λ, cone.dual_point, cone)
    W = similar(point, dim, dim)
    CO.scalmat_prod!(W, Matrix{T}(I, cone.dim, cone.dim), cone)
    @test W * λ ≈ cone.point atol=tol rtol=tol
    prod = similar(point)
    @test CO.scalmat_prod!(prod, λ, cone) ≈ cone.point atol=tol rtol=tol
    @test W * W' ≈ inv_hess atol=tol rtol=tol
    # WW' * z = W * λ
    WWz = CO.inv_hess_prod!(prod, cone.dual_point, cone)
    Wλ = CO.scalmat_prod!(prod, λ, cone)
    @test WWz ≈ Wλ atol=tol rtol=tol
    #
    # # TODO this is probably an internal function only needed for SOC
    # @test CO.scalmat_ldiv!(prod, cone.point, cone) ≈ λ atol=tol rtol=tol
    # @test CO.scalmat_ldiv!(prod_mat, W, cone) ≈ I atol=tol rtol=tol

    e1 = CO.set_initial_point(zeros(T, cone.dim), cone)
    # # e1 = W * λ \circ -grad
    @test e1 ≈ W * CO.conic_prod!(prod, cone, λ, -grad) atol=tol rtol=tol # WRONG WAY AROUND, strange that this passes
    @test e1 ≈ CO.conic_prod!(prod, cone, λ, -W * grad) atol=tol rtol=tol
    λinv = CO.scalvec_ldiv!(prod, cone, e1)
    @test CO.conic_prod!(prod, cone, λinv, λ) ≈ e1 atol=tol rtol=tol


    primal_dir = randn(cone.dim) ./ norm(cone.point) ./ 2
    dual_dir = randn(cone.dim) ./ norm(cone.dual_point) ./ 2
    max_step = CO.step_max_dist(cone, primal_dir, dual_dir)

    correction = CO.correction(cone, primal_dir, dual_dir)
    prod2 = similar(prod)
    # λ \circ W * correction = actual Mehrotra term
    @test CO.conic_prod!(prod2, cone, λ, W * correction) ≈ CO.conic_prod!(prod, cone, W \ primal_dir, W * dual_dir) atol=tol rtol=tol
    # correction = -grad * actual Mehrotra term => -grad = Winv * λinv
    # assuming this fact is false, test failing
    # @test cone.correction ≈ CO.conic_prod!(prod2, cone, -cone.grad, CO.conic_prod!(prod, cone, W \ primal_dir, W * dual_dir)) atol=tol rtol=tol

    @test CO.check_feas(cone, cone.point + 0.99 * max_step * primal_dir, true) && CO.check_feas(cone, cone.dual_point + 0.99 * max_step * dual_dir, false)
    if max_step < one(T)
        @test !CO.check_feas(cone, cone.point + 1.01 * max_step * primal_dir, true) || !CO.check_feas(cone, cone.dual_point + 1.01 * max_step * dual_dir, false)
    end

    return
end

function test_nonnegative_barrier(T::Type{<:Real})
    barrier = (s -> -sum(log, s))
    for dim in [1, 3, 6]
        test_barrier_oracles(CO.Nonnegative{T}(dim, use_scaling = false), barrier)
        test_barrier_scaling_oracles(CO.Nonnegative{T}(dim, use_scaling = true))
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
    for dim in [2, 4]
        test_barrier_oracles(CO.EpiNormEucl{T}(dim, use_scaling = false), barrier)
        test_barrier_scaling_oracles(CO.EpiNormEucl{T}(dim, use_scaling = true))
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
            CO.vec_to_mat_U!(S, s)
            return -logdet(cholesky!(Symmetric(S, :U)))
        end
        dim = div(side * (side + 1), 2)
        test_barrier_oracles(CO.PosSemidefTri{T, T}(dim), R_barrier)
        # complex PSD cone
        function C_barrier(s)
            S = zeros(Complex{eltype(s)}, side, side)
            CO.vec_to_mat_U!(S, s)
            return -logdet(cholesky!(Hermitian(S, :U)))
        end
        dim = side^2
        test_barrier_oracles(CO.PosSemidefTri{T, Complex{T}}(dim), C_barrier)
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
