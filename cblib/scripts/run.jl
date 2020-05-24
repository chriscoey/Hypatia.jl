#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors
=#

using CSV
import MathOptInterface
const MOI = MathOptInterface
const MOIB = MathOptInterface.Bridges
const MOIU = MathOptInterface.Utilities
const MOIF = MathOptInterface.FileFormats
using LinearAlgebra
import TimerOutputs
import Hypatia
const SO = Hypatia.Solvers

set = "exporthantsmall"
# set = "exporthant"
# set = "exporthantmost"
# set = "myset"
# set = "myset2"
# set = "myset3"
# set = "failing6"
# set = "failing7"
# set = "powersmall"

cblib_dir = joinpath(ENV["HOME"], "cblib/cblib.zib.de/download/all")

newT = Float64
# newT = BigFloat
# tol = sqrt(eps())
tol = 1e-8
# tol = 1e-7
options = (
    verbose = true,
    iter_limit = 100,
    time_limit = 120,
    tol_rel_opt = tol,
    tol_abs_opt = tol,
    tol_feas = tol,
    system_solver = SO.QRCholDenseSystemSolver{newT}(),
    # system_solver = SO.NaiveDenseSystemSolver{newT}(),
    )


convert_cone(cone::Hypatia.Cones.Nonnegative, out_type::Type) = Hypatia.Cones.Nonnegative{out_type}(cone.dim)
convert_cone(cone::Hypatia.Cones.HypoPerLog, out_type::Type) = Hypatia.Cones.HypoPerLog{out_type}(cone.dim)
convert_cone(cone::Hypatia.Cones.Power, out_type::Type) = Hypatia.Cones.Power{out_type}(out_type.(cone.alpha), cone.n)

setfile = joinpath(@__DIR__, "../sets", set * ".txt")
if !isfile(setfile)
    error("instance set file not found: $setfile")
end
instances = SubString[]
for l in readlines(setfile)
    str = split(strip(l))
    if !isempty(str)
        str1 = first(str)
        if !startswith(str1, '#')
            push!(instances, str1)
        end
    end
end

println("\nstarting run\n")

opt_count = 0
all_iter_counts = Int[]
opt_iter_counts = Int[]
all_times = Float64[]
opt_times = Float64[]
failed = SubString[]

for instname in instances
    println("\nstarting $instname")

    model = MOIF.CBF.Model()
    MOI.read_from_file(model, joinpath(cblib_dir, instname * ".cbf.gz"))

    cache = MOIU.UniversalFallback(MOIU.Model{Float64}());
    MOI.copy_to(cache, model);
    integer_indices = MOI.get(cache, MOI.ListOfConstraintIndices{MOI.SingleVariable, MOI.Integer}());
    MOI.delete.(cache, integer_indices);

    mock_optimizer = Hypatia.Optimizer()
    MOI.empty!(mock_optimizer)
    MOI.copy_to(mock_optimizer, cache);

    model = mock_optimizer.model
    model = Hypatia.Models.Model{newT}(
        newT.(model.c),
        newT.(model.A),
        newT.(model.b),
        newT.(model.G),
        newT.(model.h),
        convert_cone.(model.cones, newT),
        )
    # @show length(model.cones)

    solver = SO.Solver{newT}(; options...)
    SO.load(solver, model)

    optimizer = Hypatia.Optimizer{newT}()
    optimizer.model = model
    optimizer.solver = solver

    SO.solve(solver)

    status = MOI.get(optimizer, MOI.TerminationStatus())
    solvetime = MOI.get(optimizer, MOI.SolveTime())
    iters = MOI.get(optimizer, MOI.BarrierIterations())
    @show status
    if status == MOI.OPTIMAL
        global opt_count += 1
        push!(opt_times, solvetime)
        push!(opt_iter_counts, iters)
    else
        push!(failed, instname)
    end
    push!(all_times, solvetime)
    push!(all_iter_counts, iters)
end

println("\ndone...\n")
println("failed instances:")
for instname in failed
    println(instname)
end
println("\n$opt_count / $(length(instances)) optimal for tol $tol")
geomean(v::Vector) = (isempty(v) ? NaN : exp(sum(log, v) / length(v)))
println("time geomeans:\n  opt $(geomean(opt_times))\n  all $(geomean(all_times))")
println("iter geomeans:\n  opt $(geomean(opt_iter_counts))\n  all $(geomean(all_iter_counts))")
