#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified/inspired from files in https://github.com/dpapp-github/alfonso/
- https://github.com/dpapp-github/alfonso/blob/master/ChebInterval.m
- https://github.com/dpapp-github/alfonso/blob/master/PaduaSquare.m
- https://github.com/dpapp-github/alfonso/blob/master/FeketeCube.m
and Matlab files in the packages
- Chebfun http://www.chebfun.org/
- Padua2DM by M. Caliari, S. De Marchi, A. Sommariva, and M. Vianello http://profs.sci.univr.it/~caliari/software.htm
=#

# domains
abstract type InterpDomain end
sampling_region(dom::InterpDomain) = dom

# hyperrectangle/box domain
mutable struct Box <: InterpDomain
    l::Vector{Float64}
    u::Vector{Float64}
    function Box(l::Vector{Float64}, u::Vector{Float64})
        @assert length(l) == length(u)
        dom = new()
        dom.l = l
        dom.u = u
        return dom
    end
end

dimension(dom::Box) = length(dom.l)
degree(::Box) = 2

function interp_sample(dom::Box, npts::Int)
    dim = dimension(dom)
    pts = rand(npts, dim) .- 0.5
    shift = (dom.u + dom.l)/2.0
    for i in 1:npts
        pts[i,:] = pts[i,:] .* (dom.u - dom.l) + shift
    end
    return pts
end

function get_weights(dom::Box, pts::AbstractMatrix{Float64})
    g = [(pts[:,i] .- dom.l[i]) .* (dom.u[i] .- pts[:,i]) for i in 1:size(pts, 2)]
    @assert all(all(gi .>= 0.0) for gi in g)
    return g
end

# Euclidean hyperball domain
mutable struct Ball <: InterpDomain
    c::Vector{Float64}
    r::Float64
    function Ball(c::Vector{Float64}, r::Float64)
        dom = new()
        dom.c = c
        dom.r = r
        return dom
    end
end

dimension(dom::Ball) = length(dom.c)
degree(::Ball) = 2

function interp_sample(dom::Ball, npts::Int)
    dim = dimension(dom)
    pts = randn(npts, dim)
    norms = sum(abs2, pts, dims=2)
    pts .*= dom.r ./ sqrt.(norms) # scale
    norms ./= 2.0
    pts .*= sf_gamma_inc_Q.(norms, dim/2) .^ inv(dim) # sf_gamma_inc_Q is the normalized incomplete gamma function
    for i in 1:dim
        pts[:, i] .+= dom.c[i] # shift
    end
    return pts
end

function get_weights(dom::Ball, pts::AbstractMatrix{Float64})
    g = [dom.r^2 - sum((pts[j,:] - dom.c).^2) for j in 1:size(pts, 1)]
    @assert all(g .>= 0.0)
    return [g]
end

# hyperellipse domain: (x-c)'Q(x-c) \leq 1
mutable struct Ellipsoid <: InterpDomain
    c::Vector{Float64}
    Q::AbstractMatrix{Float64}
    function Ellipsoid(c::Vector{Float64}, Q::AbstractMatrix{Float64})
        @assert length(c) == size(Q, 1)
        dom = new()
        dom.c = c
        dom.Q = Q
        return dom
    end
end

dimension(dom::Ellipsoid) = length(dom.c)
degree(::Ellipsoid) = 2

function interp_sample(dom::Ellipsoid, npts::Int)
    dim = dimension(dom)
    pts = randn(npts, dim)
    norms = sum(abs2, pts, dims=2)
    for i in 1:npts
        pts[i,:] ./= sqrt(norms[i]) # scale
    end
    norms ./= 2.0
    pts .*= sf_gamma_inc_Q.(norms, dim/2) .^ inv(dim) # sf_gamma_inc_Q is the normalized incomplete gamma function

    rotscale = cholesky(dom.Q).U
    for i in 1:npts
        pts[i,:] = rotscale\pts[i,:] # rotate/scale
    end

    for i in 1:dim
        pts[:, i] .+= dom.c[i] # shift
    end
    return pts
end

function get_weights(dom::Ellipsoid, pts::AbstractMatrix{Float64})
    g = [1.0 - (pts[j, :] - dom.c)' * dom.Q * (pts[j, :] - dom.c) for j in 1:size(pts, 1)]
    @assert all(g .>= 0.0)
    return [g]
end

# assumes the free part has the same dimension as the restricted part
mutable struct SemiFreeDomain <: InterpDomain
    sampling_region::InterpDomain
end

sampling_region(dom::SemiFreeDomain) = dom.sampling_region

function add_free_vars(dom::InterpDomain)
    return SemiFreeDomain(dom)
end

dimension(dom::SemiFreeDomain) = 2*dimension(dom.sampling_region)
degree(dom::SemiFreeDomain) = degree(dom.sampling_region)

function interp_sample(dom::SemiFreeDomain, npts::Int)
    return hcat(interp_sample(dom.sampling_region, npts), interp_sample(dom.sampling_region, npts))
end

function get_weights(dom::SemiFreeDomain, pts::Matrix{Float64})
    count = div(size(pts, 2), 2)
    return get_weights(dom.sampling_region, view(pts, : ,1:count))
end

function get_LU(n::Int, d::Int)
    L = binomial(n+d,n)
    U = binomial(n+2d, n)
    return (L, U)
end

function interpolate(
    dom::InterpDomain,
    d::Int;
    sample::Bool = false,
    calc_w::Bool = false,
    sample_factor::Int = 10,
    )
    if sample
        return wsos_sample_params(dom, d, calc_w=calc_w, sample_factor=sample_factor)
    else
        return wsos_box_params(sampling_region(dom), dimension(dom), d, calc_w=calc_w)
    end
end

function wsos_box_params(dom::InterpDomain, ::Int, ::Int)
    error("accurate methods for interpolation points are only available for box domains, use sampling instead")
end

# slow but high-quality hyperrectangle/box point selections
function wsos_box_params(dom::Box, n::Int, d::Int; calc_w::Bool=false)
    # n could be larger than the dimension of dom if the original domain was a SemiFreeDomain
    if n == 1
        (U, pts, P0, P0sub, w) = cheb2_data(d, calc_w)
    elseif n == 2
        (U, pts, P0, P0sub, w) = padua_data(d, calc_w) # or approxfekete_data(n, d)
    elseif n > 2
        (U, pts, P0, P0sub, w) = approxfekete_data(n, d, calc_w)
    end
    # difference with sampling functions is that P0 is always formed using points in [-1, 1]
    # scale and shift points, get WSOS matrices
    pscale = [0.5*(dom.u[mod(j-1,dimension(dom))+1] - dom.l[mod(j-1,dimension(dom))+1]) for j in 1:n]
    pshift = [0.5*(dom.u[mod(j-1,dimension(dom))+1] + dom.l[mod(j-1,dimension(dom))+1]) for j in 1:n]
    Wtsfun = (j -> sqrt.(1.0 .- abs2.(pts[:,j]))*pscale[j])
    PWts = [Wtsfun(j) .* P0sub for j in 1:dimension(dom)]
    trpts = pts .* pscale' .+ pshift'

    return (U=U, pts=trpts, P0=P0, PWts=PWts, w=w)
end

# return k Chebyshev points of the second kind
cheb2_pts(k::Int) = [-cospi(j/(k-1)) for j in 0:k-1]

function calc_u(n::Int, d::Int, pts::Matrix{Float64})
    u = Vector{Matrix{Float64}}(undef, n)
    for j in 1:n
        uj = u[j] = Matrix{Float64}(undef, size(pts, 1), d+1)
        uj[:,1] .= 1.0
        @. @views uj[:,2] = pts[:,j]
        for t in 3:d+1
            @. @views uj[:,t] = 2.0*uj[:,2]*uj[:,t-1] - uj[:,t-2]
        end
    end
    return u
end

function cheb2_data(d::Int, calc_w::Bool)
    @assert d > 0
    U = 2d + 1

    # Chebyshev points for degree 2d
    pts = reshape(cheb2_pts(U), :, 1)

    # evaluations
    P0 = calc_u(1, d, pts)[1]
    P0sub = view(P0, :, 1:d)

    # weights for Clenshaw-Curtis quadrature at pts
    if calc_w
        wa = Float64[2/(1 - j^2) for j in 0:2:U-1]
        append!(wa, wa[floor(Int, U/2):-1:2])
        w = real.(FFTW.ifft(wa))
        w[1] = w[1]/2
        push!(w, w[1])
    else
        w = Float64[]
    end

    return (U, pts, P0, P0sub, w)
end

function padua_data(d::Int, calc_w::Bool)
    @assert d > 0
    (L, U) = get_LU(2, d)

    # Padua points for degree 2d
    cheba = cheb2_pts(2d+1)
    chebb = cheb2_pts(2d+2)
    pts = Matrix{Float64}(undef, U, 2)
    j = 1
    for a in 0:2d
        for b in 0:2d+1
            if iseven(a+b)
                pts[j,1] = -cheba[a+1]
                pts[U+1-j,2] = -chebb[2d+2-b]
                j += 1
            end
        end
    end

    # evaluations
    u = calc_u(2, d, pts)
    P0 = Matrix{Float64}(undef, U, L)
    P0[:,1] .= 1
    col = 1
    for t in 1:d
        for xp in Combinatorics.multiexponents(2, t)
            col += 1
            P0[:,col] .= u[1][:,xp[1]+1] .* u[2][:,xp[2]+1]
        end
    end
    P0sub = view(P0, :, 1:binomial(1+d, 2))

    # cubature weights at Padua points
    # even-degree Chebyshev polynomials on the subgrids
    if calc_w
        te1 = [cospi(i*j/(2d)) for i in 0:2:2d, j in 0:2:2d]
        to1 = [cospi(i*j/(2d)) for i in 0:2:2d, j in 1:2:2d]
        te2 = [cospi(i*j/(2d+1)) for i in 0:2:2d, j in 0:2:2d+1]
        to2 = [cospi(i*j/(2d+1)) for i in 0:2:2d, j in 1:2:2d+1]
        te1[2:d+1,:] .*= sqrt(2)
        to1[2:d+1,:] .*= sqrt(2)
        te2[2:d+1,:] .*= sqrt(2)
        to2[2:d+1,:] .*= sqrt(2)
        # even, even moments matrix
        mom = 2*sqrt(2)./[1 - i^2 for i in 0:2:2d]
        mom[1] = 2
        Mmom = zeros(d+1, d+1)
        f = 1/(d*(2d+1))
        for j in 1:d+1, i in 1:d+2-j
            Mmom[i,j] = mom[i]*mom[j]*f
        end
        Mmom[1,d+1] /= 2
        # cubature weights as matrices on the subgrids
        W = Matrix{Float64}(undef, d+1, 2d+1)
        W[:,1:2:2d+1] .= to2'*Mmom*te1
        W[:,2:2:2d+1] .= te2'*Mmom*to1
        W[:,[1,2d+1]] ./= 2
        W[1,2:2:2d+1] ./= 2
        W[d+1,1:2:2d+1] ./= 2
        w = vec(W)
    else
        w = Float64[]
    end

    return (U, pts, P0, P0sub, w)
end

function approxfekete_data(n::Int, d::Int, calc_w::Bool)
    @assert d > 0
    @assert n > 1
    (L, U) = get_LU(n, d)

    # points in the initial interpolation grid
    npts = prod(2d+1:2d+n)
    candidate_pts = Matrix{Float64}(undef, npts, n)
    for j in 1:n
        ig = prod(2d+1+j:2d+n)
        cs = cheb2_pts(2d+j)
        i = 1
        l = 1
        while true
            candidate_pts[i:i+ig-1,j] .= cs[l]
            i += ig
            l += 1
            if l >= 2d+1+j
                if i >= npts
                    break
                end
                l = 1
            end
        end
    end
    dom = Box(-ones(n), ones(n))
    (pts, P0, P0sub, w) = make_wsos_arrays(dom, candidate_pts, d, U, L, calc_w=calc_w)
    return (U, pts, P0, P0sub, w)
end

function make_wsos_arrays(
    dom::InterpDomain,
    candidate_pts::Matrix{Float64},
    d::Int,
    U::Int,
    L::Int;
    calc_w::Bool = false,
    )
    (npts, n) = size(candidate_pts)
    u = calc_u(n, 2d, candidate_pts)
    m = Vector{Float64}(undef, U)
    m[1] = 2^n
    M = Matrix{Float64}(undef, npts, U)
    M[:,1] .= 1.0

    col = 1
    for t in 1:2d
        for xp in Combinatorics.multiexponents(n, t)
            col += 1
            if any(isodd, xp)
                m[col] = 0.0
            else
                m[col] = m[1]/prod(1.0 - abs2(xp[j]) for j in 1:n)
            end
            @. @views M[:,col] = u[1][:,xp[1]+1]
            for j in 2:n
                @. @views M[:,col] *= u[j][:,xp[j]+1]
            end
        end
    end

    F = qr!(Array(M'), Val(true))
    keep_pnt = F.p[1:U]
    pts = candidate_pts[keep_pnt,:] # subset of points indexed with the support of w
    P0 = M[keep_pnt, 1:L] # subset of polynomial evaluations up to total degree d

    subd = d - div(degree(dom), 2)
    P0sub = view(P0, :, 1:binomial(n+subd, n))

    if calc_w
        Qtm = F.Q'*m
        w = UpperTriangular(F.R[:,1:U])\Qtm
    else
        w = Float64[]
    end

    return (pts, P0, P0sub, w)
end

# fast, sampling-based point selection for general domains
function wsos_sample_params(
    dom::InterpDomain,
    d::Int;
    calc_w::Bool = false,
    sample_factor::Int = 10,
    )
    n = dimension(dom)
    (L, U) = get_LU(n, d)
    candidate_pts = interp_sample(dom, U * sample_factor)
    (pts, P0, P0sub, w) = make_wsos_arrays(dom, candidate_pts, d, U, L, calc_w=calc_w)
    g = get_weights(dom, pts)
    PWts = [sqrt.(gi) .* P0sub for gi in g]
    return (U=U, pts=pts, P0=P0, PWts=PWts, w=w)
end
