#=
run barrier tests
=#

using Test
using Printf
import Hypatia.Cones
include(joinpath(@__DIR__, "cone.jl"))

function cone_types(T::Type{<:Real})
    cones_T = [
        Cones.Nonnegative{T},
        Cones.PosSemidefTri{T, T},
        Cones.PosSemidefTri{T, Complex{T}},
        Cones.DoublyNonnegativeTri{T},
        Cones.PosSemidefTriSparse{Cones.PSDSparseDense, T, T},
        Cones.PosSemidefTriSparse{Cones.PSDSparseDense, T, Complex{T}},
        Cones.LinMatrixIneq{T},
        Cones.EpiNormInf{T, T},
        Cones.EpiNormInf{T, Complex{T}},
        Cones.EpiNormEucl{T},
        Cones.EpiPerSquare{T},
        Cones.EpiNormSpectral{T, T},
        Cones.EpiNormSpectral{T, Complex{T}},
        Cones.MatrixEpiPerSquare{T, T},
        Cones.MatrixEpiPerSquare{T, Complex{T}},
        Cones.GeneralizedPower{T},
        Cones.HypoPowerMean{T},
        Cones.HypoGeoMean{T},
        Cones.HypoRootdetTri{T, T},
        Cones.HypoRootdetTri{T, Complex{T}},
        Cones.HypoPerLog{T},
        Cones.HypoPerLogdetTri{T, T},
        Cones.HypoPerLogdetTri{T, Complex{T}},
        Cones.EpiPerSepSpectral{Cones.VectorCSqr{T}, T},
        Cones.EpiPerSepSpectral{Cones.MatrixCSqr{T, T}, T},
        Cones.EpiPerSepSpectral{Cones.MatrixCSqr{T, Complex{T}}, T},
        Cones.EpiRelEntropy{T},
        Cones.EpiTrRelEntropyTri{T},
        Cones.WSOSInterpNonnegative{T, T},
        Cones.WSOSInterpNonnegative{T, Complex{T}},
        Cones.WSOSInterpPosSemidefTri{T},
        Cones.WSOSInterpEpiNormEucl{T},
        Cones.WSOSInterpEpiNormOne{T},
        ]

    if T <: LinearAlgebra.BlasReal
        append!(cones_T, [
            Cones.PosSemidefTriSparse{Cones.PSDSparseCholmod, T, T},
            Cones.PosSemidefTriSparse{Cones.PSDSparseCholmod, T, Complex{T}},
            ])
    end

    return cones_T
end

sep_spectral_funs = [
    Cones.NegLogSSF(),
    Cones.NegEntropySSF(),
    Cones.NegSqrtSSF(),
    Cones.NegPower01SSF(3//10),
    Cones.Power12SSF(1.5),
    ]

@testset "cone tests" begin

println("starting oracle tests")
@testset "oracle tests" begin
real_types = [
    Float64,
    # Float32,
    # BigFloat,
    ]
@testset "$cone" for T in real_types, cone in cone_types(T)
    println("$cone")
    test_time = @elapsed test_oracles(cone)
    @printf("%8.2e seconds\n", test_time)
end
end

# println("\nstarting barrier tests")
# @testset "barrier tests" begin
# real_types = [
#     Float64,
#     # Float32,
#     # BigFloat,
#     ]
# @testset "$cone" for T in real_types, cone in cone_types(T)
#     println("$cone")
#     test_time = @elapsed test_barrier(cone)
#     @printf("%8.2e seconds\n", test_time)
# end
# end

# println("\nstarting time/allocation measurements")
# @testset "allocation tests" begin
# real_types = [
#     Float64,
#     # Float32,
#     # BigFloat,
#     ]
# @testset "$cone" for T in real_types, cone in cone_types(T)
#     println("\n$cone")
#     test_time = @elapsed show_time_alloc(cone)
#     @printf("%8.2e seconds\n", test_time)
# end
# println()
# end

end
;
