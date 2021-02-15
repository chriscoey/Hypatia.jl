
relaxed_tols = (default_tol_relax = 100,)
insts = Dict()
insts["minimal"] = [
    ((2, 3),),
    ((2, 3), SOCExpPSDOptimizer),
    ]
insts["fast"] = [
    ((5, 10), nothing, relaxed_tols),
    ((5, 10), SOCExpPSDOptimizer, relaxed_tols),
    ((10, 20), nothing, relaxed_tols),
    ((10, 20), SOCExpPSDOptimizer, relaxed_tols),
    ((20, 40), nothing, relaxed_tols),
    ((20, 40), SOCExpPSDOptimizer, relaxed_tols),
    ((40, 80), nothing, relaxed_tols),
    ((40, 80), SOCExpPSDOptimizer, relaxed_tols),
    ((100, 150), nothing, relaxed_tols),
    ((100, 150), SOCExpPSDOptimizer, relaxed_tols),
    ]
insts["slow"] = [
    ((40, 80), SOCExpPSDOptimizer, relaxed_tols),
    ((100, 200), nothing, relaxed_tols),
    ]
insts["various"] = [
    ((5, 10),),
    ((5, 10), SOCExpPSDOptimizer),
    ((10, 20),),
    ((10, 20), SOCExpPSDOptimizer),
    ((20, 40),),
    ((20, 40), SOCExpPSDOptimizer),
    ((40, 80),),
    ((40, 80), SOCExpPSDOptimizer),
    ((100, 150),),
    ((100, 150), SOCExpPSDOptimizer),
    ]
return (RobustGeomProgJuMP, insts)
