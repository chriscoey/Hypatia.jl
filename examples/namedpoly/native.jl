#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/polyOpt.m
formulates and solves the (dual of the) polynomial optimization problem for a given polynomial, described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

using Hypatia
using LinearAlgebra
using Test

function build_namedpoly(
    polyname::Symbol,
    d::Int;
    )
    # get data for named polynomial
    (n, lbs, ubs, deg, fn) = polys[polyname]
    if d < ceil(Int, deg/2)
        error("requires d >= $(ceil(Int, deg/2))")
    end

    # generate interpolation
    dom = Hypatia.Box(lbs, ubs)
    (U, pts, P0, PWts, _) = Hypatia.interpolate(dom, d, sample=false)

    # set up problem data
    A = ones(1, U)
    b = [1.0,]
    c = [fn(pts[j,:]...) for j in 1:U] # evaluate polynomial at transformed points
    G = Diagonal(-1.0I, U) # TODO uniformscaling?
    h = zeros(U)

    cone = Hypatia.Cone([Hypatia.WSOSPolyInterp(U, [P0, PWts...], true)], [1:U])

    return (c, A, b, G, h, cone)
end

function run_namedpoly()
    # select the named polynomial to minimize and the SOS degree (to be squared)
    (c, A, b, G, h, cone) =
        # build_namedpoly(:butcher, 2)
        # build_namedpoly(:caprasse, 4)
        # build_namedpoly(:goldsteinprice, 7)
        # build_namedpoly(:heart, 2)
        # build_namedpoly(:lotkavolterra, 3)
        # build_namedpoly(:magnetism7, 2)
        # build_namedpoly(:motzkin, 7)
        build_namedpoly(:reactiondiffusion, 4)
        # build_namedpoly(:robinson, 8)
        # build_namedpoly(:rosenbrock, 4)
        # build_namedpoly(:schwefel, 3)

    # Hypatia.check_data(c, A, b, G, h, cone)
    # (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c, A, b, G, useQR=true)
    # L = Hypatia.QRSymmCache(c1, A1, b1, G1, h, cone, Q2, RiQ1)

    # Hypatia.check_data(P, c, A, b, G, h, cone)

    mdl = Hypatia.Model(maxiter=200, verbose=true)
    Hypatia.load_data!(mdl, zeros(length(c), length(c)), c, A, b, G, h, cone)
    # Hypatia.load_data!(mdl, c1, A1, b1, G1, h, cone, L)
    Hypatia.solve!(mdl)

    # x = zeros(length(c))
    # x[dukeep] = Hypatia.get_x(mdl)
    # y = zeros(length(b))
    # y[prkeep] = Hypatia.get_y(mdl)
    # s = Hypatia.get_s(mdl)
    # z = Hypatia.get_z(mdl)

    status = Hypatia.get_status(mdl)
    solvetime = Hypatia.get_solvetime(mdl)
    pobj = Hypatia.get_pobj(mdl)
    dobj = Hypatia.get_dobj(mdl)

    @test status == :Optimal
    @show status
    # @show x
    @show pobj
    @show dobj
    return nothing
end

# list of currently available named polynomials, see https://people.sc.fsu.edu/~jburkardt/py_src/polynomials/polynomials.html
polys = Dict{Symbol,NamedTuple}(
    :butcher => (n=6, lbs=[-1.0,-0.1,-0.1,-1.0,-0.1,-0.1], ubs=[0.0,0.9,0.5,-0.1,-0.05,-0.03], deg=3,
        fn=((u,v,w,x,y,z) -> z*v^2+y*w^2-u*x^2+x^3+x^2-(1/3)*u+(4/3)*x)
        ),
    :caprasse => (n=4, lbs=-0.5*ones(4), ubs=0.5*ones(4), deg=8,
        fn=((w,x,y,z) -> -w*y^3+4x*y^2*z+4w*y*z^2+2x*z^3+4w*y+4y^2-10x*z-10z^2+2)
        ),
    :goldsteinprice => (n=2, lbs=-2.0*ones(2), ubs=2.0*ones(2), deg=8,
        fn=((x,y) -> (1+(x+y+1)^2*(19-14x+3x^2-14y+6x*y+3y^2))*(30+(2x-3y)^2*(18-32x+12x^2+48y-36x*y+27y^2)))
        ),
    :heart => (n=8, lbs=[-0.1,0.4,-0.7,-0.7,0.1,-0.1,-0.3,-1.1], ubs=[0.4,1,-0.4,0.4,0.2,0.2,1.1,-0.3], deg=4,
        fn=((s,t,u,v,w,x,y,z) -> s*x^3-3s*x*y^2+u*y^3-3u*y*x^2+t*w^3-3*t*w*z^2+v*z^3-3v*z*w^2+0.9563453)
        ),
    :lotkavolterra => (n=4, lbs=-2.0*ones(4), ubs=2.0*ones(4), deg=3,
        fn=((w,x,y,z) -> w*(x^2+y^2+z^2-1.1)+1)
        ),
    :magnetism7 => (n=7, lbs=-ones(7), ubs=ones(7), deg=2,
        fn=((t,u,v,w,x,y,z) -> t^2+2u^2+2v^2+2w^2+2x^2+2y^2+2z^2-t)
        ),
    :motzkin => (n=2, lbs=-ones(2), ubs=ones(2), deg=6,
        fn=((x,y) -> 1-48x^2*y^2+64x^2*y^4+64x^4*y^2)
        ),
    :reactiondiffusion => (n=3, lbs=-5.0*ones(3), ubs=5.0*ones(3), deg=2,
        fn=((x,y,z) -> -x+2y-z-0.835634534y*(1+y))
        ),
    :robinson => (n=2, lbs=-ones(2), ubs=ones(2), deg=6,
        fn=((x,y) -> 1+x^6+y^6-x^4*y^2+x^4-x^2*y^4+y^4-x^2+y^2+3x^2*y^2)
        ),
    :rosenbrock => (n=2, lbs=-5.0*ones(2), ubs=10.0*ones(2), deg=4,
        fn=((x,y) -> (1-x)^2+100*(x^2-y)^2)
        ),
    :schwefel => (n=3, lbs=-10.0*ones(3), ubs=10.0*ones(3), deg=4,
        fn=((x,y,z) -> (x-y^2)^2+(y-1)^2+(x-z^2)^2+(z-1)^2)
        ),
)

return
