#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
Copyright 2018, David Papp, Sercan Yildiz

utilities for interpolation on canonical domains

modified/inspired from files in https://github.com/dpapp-github/alfonso/
- https://github.com/dpapp-github/alfonso/blob/master/ChebInterval.m
- https://github.com/dpapp-github/alfonso/blob/master/PaduaSquare.m
- https://github.com/dpapp-github/alfonso/blob/master/FeketeCube.m
and Matlab files in the packages
- Chebfun http://www.chebfun.org/
- Padua2DM by M. Caliari, S. De Marchi, A. Sommariva, and M. Vianello http://profs.sci.univr.it/~caliari/software.htm
=#

sampling_region(dom::SemiFreeDomain) = dom.restricted_halfregion
sampling_region(dom::Domain) = dom


function interp_sample(dom::Box{T}, npts::Int) where {T <: Real}
    dim = get_dimension(dom)
    pts = rand(T, npts, dim) .- T(0.5)
    shift = (dom.u + dom.l) .* T(0.5)
    for i in 1:npts
        pts[i, :] = pts[i, :] .* (dom.u - dom.l) + shift
    end
    return pts
end

function get_weights(dom::Box{T}, pts::AbstractMatrix{T}) where {T <: Real}
    g = [(pts[:, i] .- dom.l[i]) .* (dom.u[i] .- pts[:, i]) for i in 1:size(pts, 2)]
    @assert all(all(gi .>= 0) for gi in g)
    return g
end


interp_sample(dom::Ball{T}, npts::Int) where {T <: Real} = error("cannot handle ball domains with number type $T")

function interp_sample(dom::Ball{Float64}, npts::Int)
    dim = get_dimension(dom)
    pts = randn(npts, dim)
    norms = sum(abs2, pts, dims = 2)
    pts .*= dom.r ./ sqrt.(norms) # scale
    norms .*= 0.5
    pts .*= sf_gamma_inc_Q.(norms, dim * 0.5) .^ inv(dim) # sf_gamma_inc_Q is the normalized incomplete gamma function
    for i in 1:dim
        pts[:, i] .+= dom.c[i] # shift
    end
    return pts
end

function get_weights(dom::Ball{T}, pts::AbstractMatrix{T}) where {T <: Real}
    g = [dom.r^2 - sum((pts[j, :] - dom.c) .^ 2) for j in 1:size(pts, 1)]
    @assert all(g .>= 0)
    return [g]
end

interp_sample(dom::Ellipsoid{T}, npts::Int) where {T <: Real} = error("cannot handle ellipsoid domains with number type $T")

function interp_sample(dom::Ellipsoid{Float64}, npts::Int)
    dim = get_dimension(dom)
    pts = randn(npts, dim)
    norms = sum(abs2, pts, dims = 2)
    for i in 1:npts
        pts[i, :] ./= sqrt(norms[i]) # scale
    end
    norms .*= 0.5
    pts .*= sf_gamma_inc_Q.(norms, dim * 0.5) .^ inv(dim) # sf_gamma_inc_Q is the normalized incomplete gamma function

    F_rotate_scale = cholesky(dom.Q).U
    for i in 1:npts
        pts[i, :] = F_rotate_scale \ pts[i, :] # rotate/scale
    end

    for i in 1:dim
        pts[:, i] .+= dom.c[i] # shift
    end
    return pts
end

function get_weights(dom::Ellipsoid{T}, pts::AbstractMatrix{T}) where {T <: Real}
    g = [1.0 - (pts[j, :] - dom.c)' * dom.Q * (pts[j, :] - dom.c) for j in 1:size(pts, 1)]
    @assert all(g .>= 0.0)
    return [g]
end


interp_sample(dom::SemiFreeDomain, npts::Int) =
    hcat(interp_sample(dom.restricted_halfregion, npts), interp_sample(dom.restricted_halfregion, npts))

get_weights(dom::SemiFreeDomain, pts::Matrix{Float64}) =
    get_weights(dom.restricted_halfregion, view(pts, :, 1:div(size(pts, 2), 2)))


interp_sample(dom::FreeDomain{T}, npts::Int) where {T <: Real} = interp_sample(Box{T}(-ones(T, dom.n), ones(T, dom.n)), npts)

get_weights(::FreeDomain{T}, ::AbstractMatrix{T}) where {T <: Real} = T[]


get_L_U(n::Int, d::Int) = (binomial(n + d, n), binomial(n + 2d, n))

function interpolate(
    dom::Domain{T},
    d::Int; # TODO make this 2d
    sample::Bool = false,
    calc_w::Bool = false,
    sample_factor::Int = 10,
    ) where {T <: Real}
    if sample
        return wsos_sample_params(dom, d, calc_w = calc_w, sample_factor = sample_factor)
    else
        return wsos_box_params(sampling_region(dom), get_dimension(dom), d, calc_w = calc_w)
    end
end

# slow but high-quality hyperrectangle/box point selections
wsos_box_params(dom::Domain, n::Int, d::Int; calc_w::Bool) = error("accurate methods for interpolation points are only available for box domains, use sampling instead")

# difference with sampling functions is that P0 is always formed using points in [-1, 1]
function wsos_box_params(dom::Box{T}, n::Int, d::Int; calc_w::Bool = false) where {T <: Real}
    # n could be larger than the dimension of dom if the original domain was a SemiFreeDomain
    (U, pts, P0, P0sub, w) = wsos_box_params(T, n, d, calc_w)

    # TODO refactor/cleanup below
    # scale and shift points, get WSOS matrices
    pscale = [T(0.5) * (dom.u[mod(j - 1, get_dimension(dom)) + 1] - dom.l[mod(j - 1,get_dimension(dom)) + 1]) for j in 1:n]
    pshift = [T(0.5) * (dom.u[mod(j - 1, get_dimension(dom)) + 1] + dom.l[mod(j - 1,get_dimension(dom)) + 1]) for j in 1:n]
    Wtsfun = (j -> sqrt.(one(T) .- abs2.(pts[:, j])) * pscale[j])
    PWts = [Wtsfun(j) .* P0sub for j in 1:get_dimension(dom)]
    trpts = pts .* pscale' .+ pshift'

    return (U = U, pts = trpts, P0 = P0, PWts = PWts, w = w)
end

function wsos_box_params(dom::FreeDomain{T}, n::Int, d::Int; calc_w::Bool = false) where {T <: Real}
    # n could be larger than the dimension of dom if the original domain was a SemiFreeDomain
    (U, pts, P0, P0sub, w) = wsos_box_params(T, n, d, calc_w)
    return (U = U, pts = pts, P0 = P0, PWts = [], w = w)
end

function wsos_box_params(T::DataType, n::Int, d::Int, calc_w::Bool)
    if n == 1
        return cheb2_data(T, d, calc_w)
    elseif n == 2
        return padua_data(T, d, calc_w) # or approxfekete_data(n, d)
    elseif n > 2
        return approxfekete_data(T, n, d, calc_w)
    end
end

# return k Chebyshev points of the second kind
cheb2_pts(T::DataType, k::Int) = [-cospi(T(j) / T(k - 1)) for j in 0:(k - 1)]

function calc_u(n::Int, d::Int, pts::Matrix{T}) where {T <: Real}
    @assert d > 0
    u = Vector{Matrix{T}}(undef, n)
    for j in 1:n
        uj = u[j] = Matrix{T}(undef, size(pts, 1), d + 1)
        uj[:, 1] .= one(T)
        @. @views uj[:, 2] = pts[:, j]
        for t in 3:(d + 1)
            @. @views uj[:, t] = 2 * uj[:, 2] * uj[:, t - 1] - uj[:, t - 2]
        end
    end
    return u
end

function cheb2_data(T::DataType, d::Int, calc_w::Bool)
    @assert d > 0
    U = 2d + 1

    # Chebyshev points for degree 2d
    pts = reshape(cheb2_pts(T, U), :, 1)

    # evaluations
    P0 = calc_u(1, d, pts)[1]
    P0sub = view(P0, :, 1:d)

    # weights for Clenshaw-Curtis quadrature at pts
    if calc_w
        if T == BigFloat
            # TODO we can probably avoid FFTW.ifft
            error("cannot compute quadrature weights using BigFloat without sampling")
        end
        wa = [2.0 / (1 - j^2) for j in 0:2:(U - 1)]
        append!(wa, wa[div(U, 2):-1:2])
        w = real.(FFTW.ifft(wa))
        w[1] *= 0.5
        push!(w, w[1])
    else
        w = Float64[]
    end

    return (U, pts, P0, P0sub, w)
end

function padua_data(T::DataType, d::Int, calc_w::Bool)
    @assert d > 0
    (L, U) = get_L_U(2, d)

    # Padua points for degree 2d
    cheba = cheb2_pts(T, 2d + 1)
    chebb = cheb2_pts(T, 2d + 2)
    pts = Matrix{T}(undef, U, 2)
    j = 1
    for a in 0:2d
        for b in 0:(2d + 1)
            if iseven(a + b)
                pts[j, 1] = -cheba[a + 1]
                pts[(U + 1 - j), 2] = -chebb[2d + 2 - b]
                j += 1
            end
        end
    end

    # evaluations
    u = calc_u(2, d, pts)
    P0 = Matrix{T}(undef, U, L)
    P0[:, 1] .= 1
    col = 1
    for t in 1:d
        for xp in Combinatorics.multiexponents(2, t)
            col += 1
            P0[:, col] .= u[1][:, xp[1] + 1] .* u[2][:, xp[2] + 1]
        end
    end
    P0sub = view(P0, :, 1:binomial(1 + d, 2))

    # cubature weights at Padua points
    # even-degree Chebyshev polynomials on the subgrids
    if calc_w
        te1 = [cospi(T(i * j) / T(2d)) for i in 0:2:2d, j in 0:2:2d]
        to1 = [cospi(T(i * j) / T(2d)) for i in 0:2:2d, j in 1:2:2d]
        te2 = [cospi(T(i * j) / T(2d + 1)) for i in 0:2:2d, j in 0:2:(2d + 1)]
        to2 = [cospi(T(i * j) / T(2d + 1)) for i in 0:2:2d, j in 1:2:(2d + 1)]
        te1[2:(d + 1), :] .*= sqrt(T(2))
        to1[2:(d + 1), :] .*= sqrt(T(2))
        te2[2:(d + 1), :] .*= sqrt(T(2))
        to2[2:(d + 1), :] .*= sqrt(T(2))
        # even, even moments matrix
        mom = T(2) * sqrt(T(2)) ./ [T(1 - i^2) for i in 0:2:2d]
        mom[1] = 2
        Mmom = zeros(T, d + 1, d + 1)
        f = inv(T(d * (2d + 1)))
        for j in 1:(d + 1), i in 1:(d + 2 - j)
            Mmom[i, j] = mom[i] * mom[j] * f
        end
        Mmom[1, d + 1] /= T(2)
        # cubature weights as matrices on the subgrids
        W = Matrix{T}(undef, d + 1, 2d + 1)
        W[:, 1:2:(2d + 1)] .= to2' * Mmom * te1
        W[:, 2:2:(2d + 1)] .= te2' * Mmom * to1
        W[:, [1, (2d + 1)]] ./= T(2)
        W[1, 2:2:(2d + 1)] ./= T(2)
        W[d + 1, 1:2:(2d + 1)] ./= T(2)
        w = vec(W)
    else
        w = Float64[]
    end

    return (U, pts, P0, P0sub, w)
end

function approxfekete_data(T::DataType, n::Int, d::Int, calc_w::Bool)
    @assert d > 0
    @assert n > 1
    (L, U) = get_L_U(n, d)

    # points in the initial interpolation grid
    npts = prod((2d + 1):(2d + n))
    candidate_pts = Matrix{T}(undef, npts, n)
    for j in 1:n
        ig = prod((2d + 1 + j):(2d + n))
        cs = cheb2_pts(T, 2d + j)
        i = 1
        l = 1
        while true
            candidate_pts[i:(i + ig - 1), j] .= cs[l]
            i += ig
            l += 1
            if l >= 2d + 1 + j
                if i >= npts
                    break
                end
                l = 1
            end
        end
    end
    dom = Box{T}(-ones(T, n), ones(T, n))
    (pts, P0, P0sub, w) = make_wsos_arrays(dom, candidate_pts, 2d, U, L, calc_w = calc_w)
    return (U, pts, P0, P0sub, w)
end

# indices of points to keep and quadrature weights at those points
function choose_interp_pts!(
    M::Matrix{T},
    candidate_pts::Matrix{T},
    deg::Int,
    U::Int,
    calc_w::Bool,
    ) where {T <: Real}

    n = size(candidate_pts, 2)
    u = calc_u(n, deg, candidate_pts)
    m = Vector{T}(undef, U)
    m[1] = 2^n
    M[:, 1] .= 1.0

    col = 1
    for t in 1:deg
        for xp in Combinatorics.multiexponents(n, t)
            col += 1
            if any(isodd, xp)
                m[col] = 0
            else
                m[col] = m[1] / prod(1.0 - abs2(xp[j]) for j in 1:n)
            end
            @. @views M[:, col] = u[1][:, xp[1] + 1]
            for j in 2:n
                @. @views M[:, col] *= u[j][:, xp[j] + 1]
            end
        end
    end
    F = qr!(Array(M'), Val(true))
    if calc_w
        Qtm = F.Q' * m
        w = UpperTriangular(F.R[:, 1:U]) \ Qtm
    else
        w = Float64[]
    end
    return (F.p[1:U], w)
end

function make_wsos_arrays(
    dom::Domain{T},
    candidate_pts::Matrix{T},
    deg::Int,
    U::Int,
    L::Int;
    calc_w::Bool = false,
    ) where {T <: Real}

    (npts, n) = size(candidate_pts)
    M = Matrix{T}(undef, npts, U)
    (keep_pts, w) = choose_interp_pts!(M, candidate_pts, deg, U, calc_w)
    pts = candidate_pts[keep_pts, :]
    P0 = M[keep_pts, 1:L] # subset of polynomial evaluations up to total degree d
    subd = div(deg - get_degree(dom), 2)
    P0sub = view(P0, :, 1:binomial(n + subd, n))

    return (pts, P0, P0sub, w)
end

# fast, sampling-based point selection for general domains
function wsos_sample_params(
    dom::Domain,
    d::Int;
    calc_w::Bool = false,
    sample_factor::Int = 10,
    )
    n = get_dimension(dom)
    (L, U) = get_L_U(n, d)
    candidate_pts = interp_sample(dom, U * sample_factor)
    (pts, P0, P0sub, w) = make_wsos_arrays(dom, candidate_pts, 2d, U, L, calc_w = calc_w)
    g = get_weights(dom, pts)
    PWts = [sqrt.(gi) .* P0sub for gi in g]
    return (U = U, pts = pts, P0 = P0, PWts = PWts, w = w)
end

# TODO should work without sampling too
function get_interp_pts(dom::Domain{T}, deg::Int; sample_factor::Int = 10, calc_w::Bool = false) where {T <: Real}
    n = get_dimension(dom)
    U = binomial(n + deg, n)
    candidate_pts = interp_sample(dom, U * sample_factor)
    M = Matrix{T}(undef, size(candidate_pts, 1), U)
    (keep_pts, w) = choose_interp_pts!(M, candidate_pts, deg, U, calc_w)
    return (candidate_pts[keep_pts, :], w)
end

function recover_lagrange_polys(pts::Matrix{T}, deg::Int) where {T <: Real}
    (U, n) = size(pts)
    DP.@polyvar x[1:n]
    monos = DP.monomials(x, 0:deg)
    vandermonde_inv = inv([monos[j](pts[i, :]) for i in 1:U, j in 1:U])
    lagrange_polys = [dot(vandermonde_inv[:, i], monos) for i in 1:U]
    return lagrange_polys
end

function calc_u(monovec::Vector{DynamicPolynomials.PolyVar{true}}, d::Int)
    n = length(monovec)
    u = Vector{Vector}(undef, n)
    for j in 1:n
        uj = u[j] = Vector{DP.Polynomial{true, Int64}}(undef, d + 1)
        uj[1] = DP.Monomial(1)
        uj[2] = monovec[j]
        for t in 3:(d + 1)
            uj[t] = 2.0 * uj[2] * uj[t - 1] - uj[t - 2]
        end
    end
    return u
end

# returns the multivariate Chebyshev polynomials in x up to degree d
function get_chebyshev_polys(x::Vector{DynamicPolynomials.PolyVar{true}}, d::Int)
    n = length(x)
    u = calc_u(x, d)
    L = binomial(n + d, n)
    M = Vector{DP.Polynomial{true, Int64}}(undef, L)
    M[1] = DP.Monomial(1)
    col = 1
    for t in 1:d, xp in Combinatorics.multiexponents(n, t)
        col += 1
        M[col] = u[1][xp[1] + 1]
        for j in 2:n
            M[col] *= u[j][xp[j] + 1]
        end
    end
    return M
end
