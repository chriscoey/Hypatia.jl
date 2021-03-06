
covarianceest_insts(ext::MatSpecExt) = [
    [(d, ext)
    for d in vcat(3, 20:20:200)] # includes compile run
    ]

insts = OrderedDict()
insts["logdet"] = (nothing, covarianceest_insts(MatLogdetCone()))
insts["sepspec"] = (nothing, covarianceest_insts(MatNegLog()))
insts["direct"] = (nothing, covarianceest_insts(MatNegLogDirect()))
insts["eigord"] = (nothing, covarianceest_insts(MatNegLogEigOrd()))
return (CovarianceEstJuMP, insts)
