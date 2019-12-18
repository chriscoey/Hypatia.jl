
#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

sets of native test instances
=#

testfuns_few = [
    nonnegative1,
    epinorminf1,
    epinorminf6,
    epinormeucl1,
    epipersquare1,
    epiperexp1,
    hypoperlog1,
    power1,
    hypogeomean1,
    epinormspectral1,
    possemideftri1,
    possemideftri5,
    hypoperlogdettri1,
    hyporootdettri1,
    wsosinterpnonnegative1,
    wsosinterppossemideftri1,
    wsosinterpepinormeucl1,
    primalinfeas1,
    primalinfeas2,
    primalinfeas3,
    dualinfeas1,
    dualinfeas2,
    dualinfeas3,
    ]

testfuns_many = [
    nonnegative1,
    nonnegative2,
    nonnegative3,
    epinorminf1,
    epinorminf2,
    epinorminf3,
    epinorminf4,
    epinorminf5,
    epinorminf6,
    epinorminf7,
    # epinorminf8,
    # epinormeucl1,
    # epinormeucl2,
    # epinormeucl3,
    # epipersquare1,
    # epipersquare2,
    # epipersquare3,
    # epiperexp1,
    # epiperexp2,
    # epiperexp3,
    # epiperexp4,
    hypoperlog1,
    hypoperlog2,
    hypoperlog3,
    hypoperlog4,
    hypoperlog5,
    hypoperlog6,
    power1,
    power2,
    power3,
    power4,
    hypogeomean1,
    hypogeomean2,
    hypogeomean3,
    epinormspectral1,
    epinormspectral2,
    epinormspectral3,
    possemideftri1,
    possemideftri2,
    possemideftri3,
    possemideftri4,
    possemideftri5,
    possemideftri6,
    possemideftri7,
    hypoperlogdettri1,
    hypoperlogdettri2,
    hypoperlogdettri3,
    hyporootdettri1,
    hyporootdettri2,
    hyporootdettri3,
    wsosinterpnonnegative1,
    wsosinterpnonnegative2,
    wsosinterpnonnegative3,
    wsosinterppossemideftri1,
    wsosinterppossemideftri2,
    wsosinterppossemideftri3,
    wsosinterpepinormeucl1,
    wsosinterpepinormeucl2,
    wsosinterpepinormeucl3,
    # primalinfeas1,
    # primalinfeas2,
    # primalinfeas3,
    # dualinfeas1,
    # dualinfeas2,
    # dualinfeas3,
    ]

# TODO add more preprocessing test instances
testfuns_preproc = [
    dimension1,
    consistent1,
    inconsistent1,
    inconsistent2,
    ]

testfuns_reduce = vcat(testfuns_few, testfuns_preproc)
