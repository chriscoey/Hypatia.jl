#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

@testset "free domains" begin
    n = 5
    d = 2
    box = Hypatia.Box(-ones(n), ones(n))
    free = Hypatia.FreeDomain(n)
    (box_U, box_pts, box_P0, box_PWts, _) = Hypatia.interpolate(box, d, sample=true, sample_factor=200)
    (free_U, free_pts, free_P0, free_PWts, _) = Hypatia.interpolate(free, d, sample=true, sample_factor=200)
    @test box_U == free_U
    @test size(box_pts) == size(free_pts)
    @test size(box_P0) == size(free_P0)
    @test norm(box_P0) ≈ norm(free_P0) atol=1.0
    @test isempty(free_PWts)

    (box_U, box_pts, box_P0, box_PWts, _) = Hypatia.interpolate(box, d, sample=false)
    (free_U, free_pts, free_P0, free_PWts, _) = Hypatia.interpolate(free, d, sample=false)
    @test box_U == free_U
    @test size(box_pts) == size(free_pts)
    @test norm(box_P0) ≈ norm(free_P0)
    @test isempty(free_PWts)
end
