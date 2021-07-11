
relaxed_tols = (default_tol_relax = 1000,)
insts = OrderedDict()
insts["minimal"] = [
    ((2, [VecNegLog(), VecNegGeom(), VecNeg2SqrtEF()]),),
    ((2, [VecNegLogEF(), VecInv(), VecNegEntropyEF()]),),
    ((2, [VecNegEntropy(), VecNeg2Sqrt()]),),
    ((2, [VecNegLogEF(), VecNegGeomEFExp()]),),
    ((3, [VecPower12(1.5), VecNegGeomEFPow()]),),
    ((2, [VecPower12EF(1.5), VecInvEF()]),),
    ]
insts["fast"] = [
    ((1000, [VecNegLog()]),),
    ((200, [VecNegLogEF()]), nothing, relaxed_tols),
    ((1000, [VecNegEntropy()]),),
    ((200, [VecNegLogEF()]), nothing, relaxed_tols),
    ((1000, [VecPower12(1.5)]), nothing, relaxed_tols),
    ]
insts["various"] = [
    ((1000, [VecNegLog()]),),
    ((7000, [VecNegEntropy()]),),
    ((500, [VecNegLogEF()]), nothing, relaxed_tols),
    ((1000, [VecNegLogEF()]), nothing, relaxed_tols),
    ((1000, [VecPower12(1.5)]),),
    ((500, [VecPower12EF(1.5)]), nothing, relaxed_tols),
    ((2000, [VecNegLog()]), nothing, relaxed_tols),
    ((100, [VecNegLogEF()]), nothing, relaxed_tols),
    ((400, [VecNegEntropy()]), nothing, relaxed_tols),
    ((4000, [VecNegEntropy()]), nothing, relaxed_tols),
    ]
return (NonparametricDistrJuMP, insts)
