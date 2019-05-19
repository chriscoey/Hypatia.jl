#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

import Random

function pass_through_cone(cone::CO.Cone, num_checks::Int)
    CO.setup_data(cone)
    for _ in 1:num_checks
        point = zeros(CO.dimension(cone))
        CO.set_initial_point(point, cone)
        CO.load_point(cone, point)

        @test CO.check_in_cone(cone)
        @test -dot(cone.point, cone.g) ≈ CO.get_nu(cone) atol=1e-9 rtol=1e-9
        @test Symmetric(cone.H, :U) * cone.point ≈ -cone.g atol=1e-9 rtol=1e-9
    end
    return
end

function test_epinormeucl_barrier()
    for dim in [2, 3, 5]
        cone = CO.EpiNormEucl(dim)
        pass_through_cone(cone, 1)
    end
    return
end

function test_epinorinf_barrier()
    for dim in [3, 5, 7]
        cone = CO.EpiNormInf(dim)
        pass_through_cone(cone, 1)
    end
    return
end

function test_epinormspectral_barrier()
    for (n, m) in [(1, 3), (2, 4)]
        cone = CO.EpiNormSpectral(n, m)
        pass_through_cone(cone, 1)
    end
    return
end

function test_epiperpower_barrier()
    for alpha in [1.5, 2.5]
        cone = CO.EpiPerPower(alpha)
        pass_through_cone(cone, 1)
    end
    return
end

function test_epipersquare_barrier()
    for dim in [3, 5, 7]
        cone = CO.EpiPerSquare(dim)
        pass_through_cone(cone, 1)
    end
    return
end

function test_epipersumexp_barrier()
    for dim in [3, 5, 7]
        cone = CO.EpiPerSumExp(dim)
        pass_through_cone(cone, 1)
    end
    return
end

function test_hypogeomean_barrier()
    Random.seed!(1)
    for dim in [3, 5, 7]
        alpha = rand(dim - 1)
        alpha ./= sum(alpha)
        cone = CO.HypoGeomean(alpha)
        pass_through_cone(cone, 1)
    end
    return
end

function test_hypoperlog_barrier()
    cone = CO.HypoPerLog()
    pass_through_cone(cone, 1)
    return
end

function test_hypoperlogdet_barrier()
    for dim in [3, 5, 8]
        cone = CO.HypoPerLogdet(dim)
        pass_through_cone(cone, 1)
    end
    return
end

function test_semidefinite_barrier()
    for dim in [1, 3, 6]
        cone = CO.PosSemidef{Float64}(dim) # real
        pass_through_cone(cone, 1)
    end
    for dim in [1, 4, 9]
        cone = CO.PosSemidef{ComplexF64}(dim) # complex
        pass_through_cone(cone, 1)
    end
    return
end

# TODO also test complex
function test_wsospolyinterp_barrier()
    Random.seed!(1)
    for n in 1:3, halfdeg in 1:3
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
        cone = CO.WSOSPolyInterp(U, [P0], true)
        pass_through_cone(cone, 1)
    end
    return
end

function test_wsospolyinterpmat_barrier()
    Random.seed!(1)
    for n in 1:3, halfdeg in 1:3, R in 1:3
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
        cone = CO.WSOSPolyInterpMat(R, U, [P0], true)
        pass_through_cone(cone, 1)
    end
    return
end

function test_wsospolyinterpsoc_barrier()
    Random.seed!(1)
    for n in 1:2, halfdeg in 1:2, R in 3:3
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
        cone = CO.WSOSPolyInterpSOC(R, U, [P0], true)
        pass_through_cone(cone, 1)
    end
    return
end
