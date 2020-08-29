#=
run comprehensive list of tests
=#

using Test

@info("starting all tests")
@testset "all tests" begin
    include(joinpath(@__DIR__, "runmodelutilitiestests.jl"))
    include(joinpath(@__DIR__, "runbarriertests.jl"))
    include(joinpath(@__DIR__, "runnativetests.jl"))
    include(joinpath(@__DIR__, "runmoitests.jl"))
    include(joinpath(@__DIR__, "runexamplestests.jl"))

    # require optional dependencies:
    # TODO maybe only run it if Pkg says the right dependency is installed
    include(joinpath(@__DIR__, "runpardisotests.jl"))
    include(joinpath(@__DIR__, "runhsltests.jl"))
end
