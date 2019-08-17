#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using Test
import Random
using LinearAlgebra
import DynamicPolynomials
import Hypatia
const MU = Hypatia.ModelUtilities

function fekete_sample()
    Random.seed!(1)
    n = 3
    halfdeg = 2
    box = MU.Box(-ones(n), ones(n))
    free = MU.FreeDomain(n)

    for sample in (true, false)
        (box_U, box_pts, box_P0, box_PWts, _) = MU.interpolate(box, halfdeg, sample = sample, sample_factor = 20)
        (free_U, free_pts, free_P0, free_PWts, _) = MU.interpolate(free, halfdeg, sample = sample, sample_factor = 20)
        @test isempty(free_PWts)
        @test box_U == free_U
        @test size(box_pts) == size(free_pts)
        @test size(box_P0) == size(free_P0)
        @test norm(box_P0) ≈ norm(free_P0) atol = 1e-1 rtol = 1e-1
    end
end

function test_recover_lagrange_polys()
    Random.seed!(1)
    n = 1
    deg = 1
    pts = reshape(Float64[0, 1], 2, 1)
    lagrange_polys = MU.recover_lagrange_polys(pts, deg)

    random_pts = rand(5)
    @test lagrange_polys[1].(random_pts) ≈ 1 .- random_pts
    @test lagrange_polys[2].(random_pts) ≈ random_pts

    deg = 2
    pts = reshape(Float64[0, 1, 2], 3, 1)
    lagrange_polys = MU.recover_lagrange_polys(pts, deg)

    random_pts = rand(5)
    @test lagrange_polys[1].(random_pts) ≈ (random_pts .- 1) .* (random_pts .- 2) * 0.5
    @test lagrange_polys[2].(random_pts) ≈ random_pts .* (random_pts .- 2) * -1
    @test lagrange_polys[3].(random_pts) ≈ random_pts .* (random_pts .- 1) * 0.5

    n = 2
    deg = 2
    pts = rand(6, 2)
    lagrange_polys = MU.recover_lagrange_polys(pts, deg)

    for i in 1:6, j in 1:6
        @test lagrange_polys[i](pts[j, :]) ≈ (j == i ? 1 : 0) atol = 1e-9
    end

    for n in 1:3, sample in [true, false]
        halfdeg = 2
        (U, pts, P0, PWts, w) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = sample, calc_w = true)
        DynamicPolynomials.@polyvar x[1:n]
        monos = DynamicPolynomials.monomials(x, 0:(2 * halfdeg))
        lagrange_polys = MU.recover_lagrange_polys(pts, 2 * halfdeg)

        @test sum(lagrange_polys) ≈ 1
        @test sum(w[i] * lagrange_polys[j](pts[i, :]) for j in 1:U, i in 1:U) ≈ sum(w)
        @test sum(w) ≈ 2^n
    end
end

function test_recover_cheb_polys()
    DynamicPolynomials.@polyvar x[1:2]
    halfdeg = 2
    monos = DynamicPolynomials.monomials(x, 0:halfdeg)
    cheb_polys = MU.get_chebyshev_polys(x, halfdeg)
    @test cheb_polys == [1, x[1], x[2], 2x[1]^2 - 1, x[1] * x[2], 2x[2]^2 - 1]
end
