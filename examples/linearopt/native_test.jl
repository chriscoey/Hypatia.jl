
insts = Dict()
insts["minimal"] = [
    ((2, 4, 1.0),),
    ((2, 4, 0.5),),
    ]
insts["fast"] = [
    ((15, 20, 1.0),),
    ((15, 20, 0.25),),
    ((100, 100, 1.0),),
    ((100, 100, 0.15),),
    ((500, 100, 1.0),),
    ((500, 100, 0.15),),
    ]
insts["slow"] = [
    ((500, 1000, 0.05),),
    ((500, 1000, 1.0),),
    ]
return (LinearOptNative, insts)
