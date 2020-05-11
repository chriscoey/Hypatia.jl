# import Hypatia
# const CO = Hypatia.Cones

# function calc_sparsity(solver, outputfile, instname)
#     open(outputfile) do fio
#         nnzA = nnz(solver.orig_model.A)
#         nnzG = nnz(solver.orig_model.G)
#         nnzQ = nnz(sparse(solver.Ap_Q))
#         nnzR = nnz(sparse(solver.Ap_R))
#         nnzQ1 = nnz(sparse(solver.Ap_Q[:, 1:solver.orig_model.p]))
#         nnzQ2 = nnz(sparse(solver.Ap_Q[:, (solver.orig_model.p + 1):end]))
#         nnzGQ1 = nnz(sparse(GQ1))
#         nnzGQ2 = nnz(sparse(GQ2))
#         nnzQ2GHGQ2 = nnz(sparse(Q2GHGQ2))
#         nnzH = sum(nnz(sparse(c.hess)) for c in cones)
#         println(fio, "$(instname),$(nnzA),$(nnzG),$(nnzQ1),$(nnzQ2),$(nnzQ1),$(nnzQ2),$(nnzGQ1),$(nnzGQ2),$(nnzQ2GHGQ2),$(nnzH)")
#         flush(fio)
#     end
# end
#

using SparseArrays
import Hypatia
const CO = Hypatia.Cones
import MathOptInterface
const MOI = MathOptInterface
using TimerOutputs
include(joinpath(@__DIR__(), "single_moi.jl"))
include(joinpath(@__DIR__(), "read_instances.jl"))

function calc_sparsity(set, outputpath)

    fio = open(joinpath(outputpath, "SPARSTIES_$set.csv"), "a")
    println(fio, "instname,nzA,nzG,nzQ,nzGQ1,nzGQ2,nzQGHGQ,nzH,qrtime")
    flush(fio)

    setfile = joinpath(@__DIR__, "../sets", set * ".txt")
    for instname in read_instances(setfile)
        println()
        println(instname)

        cbf_model = read_model(instname, false)
        optimizer = Hypatia.Optimizer{Float64}(use_dense = false, load_only = true)
        MOI.copy_to(optimizer, cbf_model)
        # (_, optimizer, _) = setup_optimizer(cbf_model, "hypatia", "qrchol_dense", false)
        # optimizer.optimizer.load_only = true
        # println("1")
        # MOI.optimize!(optimizer)
        println("2")
        orig_model = optimizer.model

        println("3")
        solver = SO.Solver{Float64}()
        solver.timer = TimerOutput()
        try
            println("4")
            solver.orig_model = orig_model
            println("5")
            SO.load(solver, orig_model)
            println("6")
            solver.point = SO.initialize_cone_point(orig_model.cones, orig_model.cone_idxs)
            println("7")
            SO.preprocess_find_initial_point(solver)
            println("8")

            model = solver.model
            nnzA = nnz(model.A) / length(model.A)
            nnzG = nnz(model.G) / length(model.G)
            println("9")

            # precompile qr
            qr(sparse(randn(3, 3)))
            if iszero(model.p)
                qrtime = 0
                nnzQ = 0
                nnzR = 0
                GQ = Matrix(model.G)
            else
                println("10")
                # qrtime = solver.timer["lsqrAp"]
                qrtime = @elapsed f = qr(sparse(model.A'))
                println("11")
                nnzQ = norm(f.Q * Matrix{Float64}(I, model.n, model.p), 0) / length(f.Q)
                println("12")
                nnzR = nnz(sparse(f.R)) / length(f.R)
                println("13")
                GQ = Matrix(model.G) * f.Q
            end
            println("14")
            GQ2 = GQ[:, (model.p + 1):end]
            nnzGQ1 = nnz(sparse(GQ[:, 1:model.p])) / length(GQ[:, 1:model.p])
            nnzGQ2 = nnz(sparse(GQ2)) / length(GQ2)

            H = zeros(model.q, model.q)
            offset = 1
            nnzH = 0
            for cone_k in model.cones
                dim = CO.dimension(cone_k)
                idxs = offset:(offset + dim - 1)
                if isa(cone_k, CO.OrthantCone)
                    nnzH += dim
                    H[idxs, idxs] .= Diagonal(randn(dim))
                else
                    H[idxs, idxs] .= Symmetric(randn(dim, dim))
                    nnzH += dim^2
                end
                offset += dim
            end
            nnzH /= length(H)
            println("10")
            Q2GHGQ2 = GQ2' * H * GQ2
            nnzQ2GHGQ2 = nnz(sparse(Q2GHGQ2)) / length(Q2GHGQ2)

            println(fio, "$(instname),$(nnzA),$(nnzG),$(nnzQ),$(nnzGQ1),$(nnzGQ2),$(nnzQ2GHGQ2),$(nnzH),$(qrtime)")
            flush(fio)
        catch e
            print(e)
        end
    end
    close(fio)
end

calc_sparsity("cbf_many", "many")
