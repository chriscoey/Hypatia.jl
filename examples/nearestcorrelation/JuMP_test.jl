
insts = Dict()
insts["minimal"] = [
    ((2,),),
    ]
insts["fast"] = [
    ((5,),),
    ((10,),),
    ]
insts["slow"] = [
    ((15,),),
    ]
insts["various"] = [
    ((5,),),
    ((10,),),
    ((15,),),
    ]
return (NearestCorrelationJuMP, insts)
