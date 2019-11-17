#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

sets of native test instances
=#

testfuns_few = [
    nonnegative1,
    epinorminf1,
    epinormeucl1,
    epipersquare1,
    epiperexp1,
    epipersumexp1,
    hypopersumlog1,
    power1,
    hypogeomean1,
    epinormspectral1,
    possemideftri1,
    possemideftricomplex1,
    hypoperlogdettri1,
    primalinfeas1,
    primalinfeas2,
    primalinfeas3,
    dualinfeas1,
    dualinfeas2,
    dualinfeas3,
    ]

# TODO check all native instances are in this list
testfuns_many = [
    # nonnegative1,
    # nonnegative2,
    # nonnegative3,
    # epinorminf1,
    # epinorminf2,
    # epinorminf3,
    # epinorminf4,
    # epinorminf5,
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
    # epipersumexp1,
    # epipersumexp2, # TODO another epipersumexp test
    # hypopersumlog1,
    # hypopersumlog2,
    # hypopersumlog3,
    # hypopersumlog4,
    # hypopersumlog5,
    # hypopersumlog6,
    # power1,
    # power2,
    # power3,
    # power4,
    hypogeomean1,
    hypogeomean2,
    hypogeomean3,
    # epinormspectral1,
    # epinormspectral2,
    # epinormspectral3,
    # possemideftri1,
    # possemideftri2,
    # possemideftri3,
    # possemideftri4,
    # possemideftricomplex1,
    # possemideftricomplex2,
    # possemideftricomplex3,
    # hypoperlogdettri1,
    # hypoperlogdettri2,
    # hypoperlogdettri3,
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
