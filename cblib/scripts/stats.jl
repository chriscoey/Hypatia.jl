# generate stats file for an instance set

# julia scripts/stats.jl cbf_many

# import Pkg
# Pkg.activate(".")

import MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities
import Hypatia
const CO = Hypatia.Cones
import MathOptFormat
using SparseArrays
using DataStructures
include(joinpath(@__DIR__, "read_instances.jl"))

println()
if length(ARGS) != 1
    error("usage: julia stats.jl set")
end

set = ARGS[1]
setfile = joinpath(@__DIR__, "../sets", set * ".txt")
instances = read_instances(setfile)
inputpath = joinpath(@__DIR__, "../instances")

statsf = open(joinpath(@__DIR__(), "../stats", "STATS_" * set * ".csv"), "w")
cblib_dir = "/home/hypatia/cblib/cblib.zib.de/download/all"

hypatia_conetypes = [
    CO.EpiNormEucl,
    CO.EpiNormInf,
    CO.EpiNormSpectral,
    CO.EpiPerExp,
    CO.EpiPerSquare,
    CO.HypoGeomean,
    CO.HypoPerLog,
    CO.HypoPerLogdetTri,
    CO.Nonnegative,
    CO.Nonpositive,
    CO.PosSemidefTri{<:Real, <:Real},
    CO.PosSemidefTri{<:Real, <:Complex},
    CO.Power,
    CO.WSOSPolyInterp{<:Real, <:Real},
    CO.WSOSPolyInterp{<:Real, <:Complex},
    ]

println(statsf, "instname,n,p,q,nonzeroA,nonzeroG,cbfbytes," *
    # total dimension for each cone type
    "epinormeucldim,epinorminfdim,epinormspectraldim,epiperexpdim," *
    "epipersquaredim,hypogeomeandim,hypoperlogdim,hypoperlogdettridim," *
    "nonnegativedim,nonpositivedim,semidefiniterealdim,semidefinitecomplexdim," *
    "powerdim,wsospolyinterprealdim,wsospolyinterpcomplexdim," *
    # number of each cone type
    "epinormeucl,epinorminf,epinormspectral," *
    "epiperexp,epipersquare,hypogeomean,hypoperlog,hypoperlogdettri," *
    "nonnegative,nonpositive,semidefinitereal,semidefinitecomplex,power,wsospolyinterpreal," *
    "wsospolyinterpcomplex,isdiscrete"
    )

cache = MOIU.UniversalFallback(MOIU.Model{Float64}());
hyp_opt = Hypatia.Optimizer{Float64}(load_only = true);

for instname in instances
    println("reading instance: $instname")
    fullpathin = joinpath(cblib_dir, instname * ".cbf.gz")

    MOI.empty!(cache)
    MOI.empty!(hyp_opt)
    model = MathOptFormat.read_from_file(fullpathin)
    MOI.copy_to(cache, model);
    integer_indices = MOI.get(cache, MOI.ListOfConstraintIndices{MOI.SingleVariable, MOI.Integer}());
    isdiscrete = !isempty(integer_indices)
    optimizer = MOIU.CachingOptimizer(MOIU.UniversalFallback(MOIU.Model{Float64}()), hyp_opt);
    MOI.copy_to(optimizer, cache)
    MOI.optimize!(optimizer)
    model = optimizer.optimizer.model
    (c, A, b, G, h, cones) = (model.c, model.A, model.b, model.G, model.h, model.cones)

    n = length(c)
    p = length(b)
    q = length(h)
    nnzA = nnz(A)
    nnzG = nnz(G)
    file_b = stat(fullpathin).size

    cone_dims = DefaultDict{Type{<:CO.Cone}, Int}(0)
    cone_counts = DefaultDict{Type{<:CO.Cone}, Int}(0)
    for (i, c) in enumerate(cones)
        for hc in hypatia_conetypes
            if isa(c, hc)
                cone_dims[hc] += CO.dimension(c)
                cone_counts[hc] += 1
                break
            end
        end
    end
    cone_dim_res = [",$(cone_dims[c])" for c in hypatia_conetypes]
    cone_count_res = [",$(cone_counts[c])" for c in hypatia_conetypes]

    println(statsf, "$instname,$n,$p,$q,$nnzA,$nnzG,$file_b$(cone_dim_res...)$(cone_count_res...),$isdiscrete")

end

close(statsf)

println("\ndone\n")
