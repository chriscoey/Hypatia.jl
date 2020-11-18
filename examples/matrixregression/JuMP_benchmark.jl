
matrixregression_insts = [
    # [(ceil(Int, 6m), m, 5m, 0, 0.2, 0, 0, 0) for m in vcat(3, 5:5:55)] # includes compile run
    # [(n, 5n, 5n, 0, 0, 0.2, 0, 0) for n in (2, 10)]#vcat(3, 5:5:15)] # includes compile run
    [(n, 5n, n, 0.1, 0, 0, 0, 0) for n in (2, 40)]#vcat(3, 5:5:15)] # includes compile run
    ]

insts = Dict()
insts["nat"] = (nothing, matrixregression_insts)
insts["ext"] = (SOCExpPSDOptimizer, matrixregression_insts)
return (MatrixRegressionJuMP, insts)
