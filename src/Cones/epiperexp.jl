#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

(closure of) epigraph of perspective of exponential (AKA exponential cone)
(u in R_+, v in R_+, w in R) : u >= v*exp(w/v)

self-concordant barrier from "A homogeneous interior-point algorithm for nonsymmetric convex conic optimization" by Skajaa & Ye 2014
-log(v*log(u/v) - w) - log(u) - log(v)

TODO
use StaticArrays
=#

mutable struct EpiPerExp{T <: Real} <: Cone{T}
    use_3order_corr::Bool
    use_dual::Bool
    point::Vector{T}
    # dual_point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    luv::T
    vluvw::T
    g1a::T
    g2a::T
    correction::Vector{T}

    function EpiPerExp{T}(
        is_dual::Bool;
        use_3order_corr::Bool = true,
        ) where {T <: Real}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.use_3order_corr = use_3order_corr
        return cone
    end
end

EpiPerExp{T}() where {T <: Real} = EpiPerExp{T}(false)

dimension(cone::EpiPerExp) = 3

use_3order_corr(cone::EpiPerExp) = cone.use_3order_corr # TODO remove from here and just use one in Cones.jl when all cones allow

reset_data(cone::EpiPerExp) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::EpiPerExp{T}) where {T <: Real}
    reset_data(cone)
    cone.point = zeros(T, 3)
    # cone.dual_point = zeros(T, 3)
    cone.grad = zeros(T, 3)
    cone.hess = Symmetric(zeros(T, 3, 3), :U)
    cone.inv_hess = Symmetric(zeros(T, 3, 3), :U)
    cone.correction = zeros(T, 3)
    return
end

get_nu(cone::EpiPerExp) = 3

function set_initial_point(arr::AbstractVector, cone::EpiPerExp)
    arr .= (1.290928, 0.805102, -0.827838) # from MOSEK paper
    return arr
end

function update_feas(cone::EpiPerExp)
    @assert !cone.feas_updated
    (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])

    if u > 0 && v > 0
        cone.luv = log(u / v)
        cone.vluvw = v * cone.luv - w
        cone.is_feas = (cone.vluvw > 0)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiPerExp)
    @assert cone.is_feas
    (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])
    vluvw = cone.vluvw

    cone.g1a = -v / u / vluvw
    cone.grad[1] = cone.g1a - inv(u)
    cone.g2a = (1 - cone.luv) / vluvw
    cone.grad[2] = cone.g2a - inv(v)
    cone.grad[3] = inv(vluvw)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiPerExp)
    @assert cone.grad_updated
    (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])
    H = cone.hess.data
    vluvw = cone.vluvw
    g1a = cone.g1a
    g2a = cone.g2a

    H[1, 3] = g1a / vluvw
    H[2, 3] = g2a / vluvw
    H[3, 3] = abs2(cone.grad[3])
    H[1, 1] = abs2(g1a) - cone.grad[1] / u
    H[1, 2] = -(v * cone.g2a + 1) / cone.vluvw / u
    H[2, 2] = abs2(g2a) + (inv(vluvw) + inv(v)) / v

    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::EpiPerExp)
    @assert cone.is_feas
    (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])
    Hi = cone.inv_hess.data
    vluvw = cone.vluvw
    vluv = vluvw + w
    denom = vluvw + 2 * v
    uvdenom = u * v / denom

    Hi[1, 1] = u * (vluvw + v) / denom * u
    Hi[2, 2] = v * (vluvw + v) / denom * v
    Hi[3, 3] = 2 * (abs2(vluv - v) + vluv * (v - w)) + abs2(w) - v / denom * abs2(vluv - 2 * v)
    Hi[1, 2] = uvdenom * v
    Hi[1, 3] = uvdenom * (2 * vluv - w)
    Hi[2, 3] = (abs2(vluv) + w * (v - vluv)) / denom * v

    cone.inv_hess_updated = true
    return cone.inv_hess
end

update_inv_hess_prod(cone::EpiPerExp) = (cone.inv_hess_updated ? nothing : update_inv_hess(cone))

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerExp)
    update_inv_hess_prod(cone)
    mul!(prod, cone.inv_hess, arr)
    return prod
end

# directional third derivative term from MOSEK paper
# TODO make efficient and improve numerics, reuse values stored in cone fields
function correction(cone::EpiPerExp{T}, s_sol::AbstractVector{T}, z_sol::AbstractVector{T}) where {T}
    update_hess(cone)
    update_inv_hess_prod(cone)
    corr = cone.correction
    (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])

    Hi_z = similar(z_sol) # TODO prealloc
    inv_hess_prod!(Hi_z, z_sol, cone)

    # -log(v * log(u / v) - w) part
    ψ = cone.vluvw
    ψp = T[v / u, cone.luv - 1, -one(T)]
    gpp = Symmetric(cone.hess - Diagonal(T[abs2(inv(u)), abs2(inv(v)), zero(T)]), :U) # TODO improve
    zz3ψp = (z_sol[1:2] + z_sol[3] * ψp[1:2]) / (ψ + 2 * v)
    ψpp_Hi_z = T[v * (zz3ψp[2] * v / u - zz3ψp[1]), u * zz3ψp[1] - v * zz3ψp[2], zero(T)]

    # term1
    corr .= z_sol[3] * 2 * ψ * gpp * s_sol
    # term2
    corr[1] += z_sol[3] * (-v * s_sol[1] / u + s_sol[2]) / u
    corr[2] += z_sol[3] * (s_sol[1] / u - s_sol[2] / v)
    # term3
    corr[1] += ((2 * v / u * Hi_z[1] - Hi_z[2]) / u * -s_sol[1] / u + Hi_z[1] / u * s_sol[2] / u) / ψ
    corr[2] += (Hi_z[1] / u * s_sol[1] / u - Hi_z[2] / v * s_sol[2] / v) / ψ
    # term4
    corr += (ψpp_Hi_z * dot(ψp, s_sol) + ψp * dot(ψpp_Hi_z, s_sol)) / ψ

    # scale
    corr /= -2

    # - log(u) - log(v) part
    corr[1] += Hi_z[1] / u * s_sol[1] / u / u
    corr[2] += Hi_z[2] / v * s_sol[2] / v / v

    return corr
end

# TODO nonsymmetric scaling helpers

# cone.barrier = (x -> -log(x[2] * log(x[1] / x[2]) - x[3]) - log(x[1]) - log(x[2]))
# cone.check_feas = (x -> (x[1] > 0) && (x[2] > 0) && (x[2] * log(x[1] / x[2]) > x[3]))

# function scalmat_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, mu::T, cone::EpiPerExp{T}) where {T}
#     if !cone.scal_hess_updated
#         update_scaling(cone, mu)
#     end
#     point = cone.point
#     dual_point = cone.dual_point
#     H_bfgs = cone.hess
#     dual_gap = cone.dual_gap
#     primal_gap = cone.primal_gap
#     ZZt = H_bfgs - dual_point * dual_point' / dot(point, dual_point) - dual_gap * dual_gap' / dot(primal_gap, dual_gap)
#     # TODO work this out analytically rather than call eig
#     f = eigen(ZZt)
#     @assert f.values[1] ≈ 0 && f.values[2] ≈ 0
#     z = f.vectors[:, 3] * sqrt(f.values[3])
#     W = zeros(T, 3, 3)
#     W[:, 1] = dual_point / sqrt(dot(point, dual_point))
#     W[:, 2] = dual_gap / sqrt(dot(primal_gap, dual_gap))
#     W[:, 3] = z
#     prod = W * arr
#     return prod
# end
