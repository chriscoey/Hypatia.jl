#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

(closure of) hypograph of perspective of sum of logarithms
(u in R, v in R_+, w in R_+^d) : u <= v*sum(log.(w/v))

barrier modified from "Primal-Dual Interior-Point Methods for Domain-Driven Formulations" by Karimi & Tuncel, 2019
-log(sum_i v*log(w_i/v) - u) - sum_i log(w_i) - d*log(v)
=#

mutable struct HypoPerLog{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    point::Vector{T}
    dual_point::Vector{T}
    timer::TimerOutput
    nu::Int

    scal::T

    feas_updated::Bool
    grad_updated::Bool
    dual_grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    scal_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    dual_grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    old_hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    scal_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    correction::Vector{T}

    lwv::T
    vlwvu::T
    lvwnivlwvu::T
    vwivlwvu::Vector{T}

    barrier::Function # TODO delete later

    dual_grad_inacc::Bool

    function HypoPerLog{T}(
        dim::Int;
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        # TODO delete when generalize again later
        @assert dim == 3
        @assert !use_dual
        @assert !use_heuristic_neighborhood
        # old:
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.dim = dim
        cone.nu = 1 + 2 * (dim - 2)
        cone.hess_fact_cache = hess_fact_cache
        cone.barrier = (x -> -log(x[2] * log(x[3] / x[2]) - x[1]) - log(x[3]) - log(x[2]))
        return cone
    end
end

# TODO only allocate the fields we use
function setup_data(cone::HypoPerLog{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.dual_grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.scal_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    cone.correction = zeros(T, dim)
    cone.vwivlwvu = zeros(T, dim - 2)
    cone.dual_grad_inacc = false
    return
end

use_scaling(cone::HypoPerLog) = true

use_correction(cone::HypoPerLog) = true

get_nu(cone::HypoPerLog) = cone.nu

reset_data(cone::HypoPerLog) = (cone.feas_updated = cone.grad_updated = cone.dual_grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.scal_hess_updated = cone.hess_fact_updated = false)

rescale_point(cone::HypoPerLog{T}, s::T) where {T} = (cone.point .*= s)

function set_initial_point(arr::AbstractVector, cone::HypoPerLog)
    (arr[1], arr[2], w) = get_central_ray_hypoperlog(cone.dim - 2)
    arr[3:end] .= w
    return arr
end

function update_feas(cone::HypoPerLog{T}) where {T}
    @assert !cone.feas_updated

    point = cone.point
    # @show norm(point)
    # cone.scal = norm(point) / sqrt(cone.nu)
    # cone.scal = 1.0
    # @show cone.scal
    # point ./= cone.scal
    # @show norm(point)

    # mu_cone = dot(point, cone.dual_point) / cone.nu
    # @show mu_cone

    u = point[1]
    v = point[2]
    w = view(point, 3:cone.dim)

    # if v > 0 && all(>(0), w)
    #     cone.lwv = sum(log(wi / v) for wi in w)
    #     cone.vlwvu = v * cone.lwv - u
    #     # @show cone.lwv
    #     @show cone.vlwvu
    #     cone.is_feas = (cone.vlwvu > 0)
    # else
    #     cone.is_feas = false
    # end

    if v <= eps(T) || any(<=(eps(T)), w)
        cone.is_feas = false
    else
        cone.lwv = sum(log(wi / v) for wi in w)
        cone.vlwvu = v * cone.lwv - u
        # cone.scal = cone.vlwvu
        # cone.scal = 1
        # cone.point = point / cone.scal
        cone.is_feas = (cone.vlwvu > eps(T))
    end



    cone.feas_updated = true
    return cone.is_feas
end

function update_dual_feas(cone::HypoPerLog{T}) where {T}
    @assert cone.dim == 3
    u = cone.dual_point[1]
    v = cone.dual_point[2]
    w = cone.dual_point[3]
    return u < -eps(T) && w > eps(T) && v - u - u * log(-w / u) > eps(T)
end

function update_grad(cone::HypoPerLog)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    d = length(w)
    g = cone.grad

    g[1] = inv(cone.vlwvu)
    cone.lvwnivlwvu = (d - cone.lwv) / cone.vlwvu
    g[2] = cone.lvwnivlwvu - d / v
    gden = -1 - v / cone.vlwvu
    @. g[3:end] = gden / w

    cone.grad_updated = true
    return cone.grad
end

# function update_grad(cone::HypoPerLog)
#     @assert cone.is_feas
#     u = BigFloat(cone.point[1])
#     v = BigFloat(cone.point[2])
#     w = BigFloat.(cone.point[3:cone.dim])
#     d = length(w)
#     g = cone.grad
#
#     lwv = sum(log(wi / v) for wi in w)
#     vlwvu = v * lwv - u
#     lvwnivlwvu = (d - lwv) / vlwvu
#
#     g[1] = inv(vlwvu)
#     g[2] = lvwnivlwvu - d / v
#     gden = -1 - v / vlwvu
#     @. g[3:end] = gden / w
#
#     cone.grad_updated = true
#     return cone.grad
# end

function update_hess(cone::HypoPerLog)
    @assert cone.grad_updated
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    d = length(w)
    vwivlwvu = cone.vwivlwvu
    lvwnivlwvu = cone.lvwnivlwvu
    g = cone.grad
    H = cone.hess.data

    vivlwvu = v / cone.vlwvu
    @. vwivlwvu = vivlwvu / w
    H[1, 1] = abs2(g[1])
    H[1, 2] = lvwnivlwvu / cone.vlwvu
    @. H[1, 3:end] = -vwivlwvu / cone.vlwvu
    H[2, 2] = abs2(lvwnivlwvu) + d * (g[1] + inv(v)) / v
    hden = (-v * lvwnivlwvu - 1) / cone.vlwvu
    @. H[2, 3:end] = hden / w
    @inbounds for j in 1:d
        j2 = 2 + j
        @inbounds for i in 1:j
            H[2 + i, j2] = vwivlwvu[i] * vwivlwvu[j]
        end
        H[j2, j2] -= g[j2] / w[j]
    end

    # @show norm(cone.point), maximum(abs, cone.point)
    # @show norm(cone.dual_point)
    # @show norm(H), maximum(abs, H)

    cone.old_hess = copy(cone.hess)

    cone.hess_updated = true
    return cone.hess
end

# function update_hess(cone::HypoPerLog)
#     @assert cone.grad_updated
#     u = BigFloat(cone.point[1])
#     v = BigFloat(cone.point[2])
#     w = BigFloat.(cone.point[3:cone.dim])
#     d = length(w)
#
#     lwv = sum(log(wi / v) for wi in w)
#     vlwvu = v * lwv - u
#     lvwnivlwvu = (d - lwv) / vlwvu
#     vivlwvu = v / vlwvu
#     vwivlwvu = vivlwvu ./ w
#     hden = (-v * lvwnivlwvu - 1) / vlwvu
#
#     g = zeros(BigFloat, cone.dim)
#     g[1] = inv(vlwvu)
#     g[2] = lvwnivlwvu - d / v
#     gden = -1 - v / (vlwvu)
#     @. g[3:end] = gden / w
#
#     H = cone.hess.data
#
#     H[1, 1] = abs2(g[1])
#     H[1, 2] = lvwnivlwvu / vlwvu
#     @. H[1, 3:end] = -vwivlwvu / vlwvu
#     H[2, 2] = abs2(lvwnivlwvu) + d * (g[1] + inv(v)) / v
#     @. H[2, 3:end] = hden / w
#     @inbounds for j in 1:d
#         j2 = 2 + j
#         @inbounds for i in 1:j
#             H[2 + i, j2] = vwivlwvu[i] * vwivlwvu[j]
#         end
#         H[j2, j2] -= g[j2] / w[j]
#     end
#
#     cone.hess_updated = true
#     return cone.hess
# end

# directional third derivative term
# TODO make efficient and improve numerics, reuse values stored in cone fields
function correction(
    cone::HypoPerLog{T},
    primal_dir::AbstractVector{T},
    dual_dir::AbstractVector{T},
    ) where {T <: Real}
    point = cone.point
    # corr = cone.correction

    # newT = BigFloat
    newT = T
    primal_dir = newT.(primal_dir)
    dual_dir = newT.(dual_dir)
    (u, v, w) = newT.(point)

    # # TODO compare below two ways to do inverse hess prod
    # Hi_z = similar(dual_dir) # TODO prealloc
    # update_hess_fact(cone)
    # inv_hess_prod!(Hi_z, dual_dir, cone)

    # TODO refac inv hess prod here
    lwv = log(w / v)
    vlwv = v * lwv
    vlwvu = vlwv - u
    denom = vlwvu + 2 * v
    wvdenom = w * v / denom
    vvdenom = (vlwvu + v) / denom
    Hi = zeros(newT, 3, 3)
    Hi[1, 1] = 2 * (abs2(vlwv - v) + vlwv * (v - u)) + abs2(u) - v / denom * abs2(vlwv - 2 * v)
    Hi[1, 2] = (abs2(vlwv) + u * (v - vlwv)) / denom * v
    Hi[1, 3] = wvdenom * (2 * vlwv - u)
    Hi[2, 2] = v * vvdenom * v
    Hi[2, 3] = wvdenom * v
    Hi[3, 3] = w * vvdenom * w
    Hi = Symmetric(Hi, :U)
    Hi_z = Hi * dual_dir

    # -log(v * log(w / v) - u) part
    # vlwvu = cone.vlwvu
    # vlwvup = T[-1, cone.lwv - 1, v / w]
    # gpp = Symmetric(cone.hess - Diagonal(T[0, abs2(inv(v)), abs2(inv(w))]), :U) # TODO improve
    # zz3vlwvup = (dual_dir[2:3] + dual_dir[1] * vlwvup[2:3]) / (vlwvu + 2 * v)
    # vlwvupp_Hi_z = T[0, w * zz3vlwvup[2] - v * zz3vlwvup[1], v * (zz3vlwvup[1] * v / w - zz3vlwvup[2])]

    # H = inv(Hi)
    # @show H - hess(cone)
    # H = hess(cone)
    @assert cone.hess_updated
    H = newT.(cone.old_hess)
    vlwvup = newT[-1, lwv - 1, v / w]
    gpp = Symmetric(H - Diagonal(newT[0, abs2(inv(v)), abs2(inv(w))]), :U) # TODO improve
    zz3vlwvup = (dual_dir[2:3] + dual_dir[1] * vlwvup[2:3]) / (vlwvu + 2 * v)
    vlwvupp_Hi_z = newT[0, w * zz3vlwvup[2] - v * zz3vlwvup[1], v * (zz3vlwvup[1] * v / w - zz3vlwvup[2])]


    # term1
    # corr = similar(primal_dir)
    corr = dual_dir[1] * 2 * vlwvu * gpp * primal_dir
    # term2
    corr[2] += dual_dir[1] * (primal_dir[3] / w - primal_dir[2] / v)
    corr[3] += dual_dir[1] * (-v * primal_dir[3] / w + primal_dir[2]) / w
    # term3
    corr[3] += ((2 * v / w * Hi_z[3] - Hi_z[2]) / w * -primal_dir[3] / w + Hi_z[3] / w * primal_dir[2] / w) / vlwvu
    corr[2] += (Hi_z[3] / w * primal_dir[3] / w - Hi_z[2] / v * primal_dir[2] / v) / vlwvu
    # term4
    corr += (vlwvupp_Hi_z * dot(vlwvup, primal_dir) + vlwvup * dot(vlwvupp_Hi_z, primal_dir)) / vlwvu

    # scale
    corr /= -2

    # - log(v) - log(w) part
    corr[2] += Hi_z[2] / v * primal_dir[2] / v / v
    corr[3] += Hi_z[3] / w * primal_dir[3] / w / w

    cone.correction .= corr

    return cone.correction
end


# (z + mu*g)' (mu*H)^-1 (z + mu*g)
# = (z + mu*g)' (Hiz / mu - s)
# = z'Hiz / mu + g'Hiz - z's - mu*g's
# = z'Hiz / mu - 2*s'z + mu*nu
# < mu * beta^2
function in_neighborhood_sy(cone::HypoPerLog{T}, mu::T) where {T <: Real}
    # mu_cone = dot(cone.point, cone.dual_point) / cone.nu
    # @show mu_cone / mu
    # mu = mu_cone

    # nbhd_tmp = cone.nbhd_tmp
    # g = grad(cone)
    # @. nbhd_tmp = cone.dual_point + mu * g

    (u, v, w) = cone.point

    # # TODO compare below two ways to do inverse hess prod
    # Hi_z = similar(dual_dir) # TODO prealloc
    # update_hess_fact(cone)
    # inv_hess_prod!(Hi_z, dual_dir, cone)

    # TODO refac inv hess prod here
    lwv = log(w / v)
    vlwv = v * lwv
    vlwvu = vlwv - u
    denom = vlwvu + 2 * v
    wvdenom = w * v / denom
    vvdenom = (vlwvu + v) / denom
    Hi = zeros(T, 3, 3)
    Hi[1, 1] = 2 * (abs2(vlwv - v) + vlwv * (v - u)) + abs2(u) - v / denom * abs2(vlwv - 2 * v)
    Hi[1, 2] = (abs2(vlwv) + u * (v - vlwv)) / denom * v
    Hi[1, 3] = wvdenom * (2 * vlwv - u)
    Hi[2, 2] = v * vvdenom * v
    Hi[2, 3] = wvdenom * v
    Hi[3, 3] = w * vvdenom * w
    Hi = Symmetric(Hi, :U)

    # @show norm(Hi * g + cone.point)

    # nbhdsqr = dot(nbhd_tmp, Hi * nbhd_tmp)

    z = cone.dual_point
    s = cone.point
    nbhdsqr2mu = dot(z, Hi, z) / mu - 2 * dot(s, z) + mu * cone.nu
    # @show nbhdsqr / mu - nbhdsqr2mu

    # if nbhdsqr2mu <= 0
    #     # @show nbhdsqr
    #     return false
    # end
    # nbhd = sqrt(nbhdsqr)
    # sy_eta = 0.99
    # sy_eta = 0.5
    # sy_eta = 0.99
    sy_eta = 2
    return (nbhdsqr2mu < mu * sy_eta^2)

    # @show nbhd / (mu * sy_eta)
    # return (nbhd < mu * sy_eta)
    return true
end






# TODO add hess prod, inv hess etc functions
# NOTE old EpiPerExp code below may be useful (cone vector is reversed)

# function update_feas(cone::EpiPerExp)
#     @assert !cone.feas_updated
#     (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])
#
#     if u > 0 && v > 0
#         cone.luv = log(u / v)
#         cone.vluvw = v * cone.luv - w
#         cone.is_feas = (cone.vluvw > 0)
#     else
#         cone.is_feas = false
#     end
#
#     cone.feas_updated = true
#     return cone.is_feas
# end
#
# function update_grad(cone::EpiPerExp)
#     @assert cone.is_feas
#     (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])
#     vluvw = cone.vluvw
#
#     cone.g1a = -v / u / vluvw
#     cone.grad[1] = cone.g1a - inv(u)
#     cone.g2a = (1 - cone.luv) / vluvw
#     cone.grad[2] = cone.g2a - inv(v)
#     cone.grad[3] = inv(vluvw)
#
#     cone.grad_updated = true
#     return cone.grad
# end
#
# function update_hess(cone::EpiPerExp)
#     @assert cone.grad_updated
#     (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])
#     H = cone.hess.data
#     vluvw = cone.vluvw
#     g1a = cone.g1a
#     g2a = cone.g2a
#
#     H[1, 3] = g1a / vluvw
#     H[2, 3] = g2a / vluvw
#     H[3, 3] = abs2(cone.grad[3])
#     H[1, 1] = abs2(g1a) - cone.grad[1] / u
#     H[1, 2] = -(v * cone.g2a + 1) / cone.vluvw / u
#     H[2, 2] = abs2(g2a) + (inv(vluvw) + inv(v)) / v
#
#     cone.hess_updated = true
#     return cone.hess
# end

# function update_inv_hess(cone::EpiPerExp)
#     @assert cone.is_feas
#     (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])
#     Hi = cone.inv_hess.data
#     vluvw = cone.vluvw
#     vluv = vluvw + w
#     denom = vluvw + 2 * v
#     uvdenom = u * v / denom
#
#     Hi[1, 1] = u * (vluvw + v) / denom * u
#     Hi[2, 2] = v * (vluvw + v) / denom * v
#     Hi[3, 3] = 2 * (abs2(vluv - v) + vluv * (v - w)) + abs2(w) - v / denom * abs2(vluv - 2 * v)
#     Hi[1, 2] = uvdenom * v
#     Hi[1, 3] = uvdenom * (2 * vluv - w)
#     Hi[2, 3] = (abs2(vluv) + w * (v - vluv)) / denom * v
#
#     cone.inv_hess_updated = true
#     return cone.inv_hess
# end
#
# function update_inv_hess_prod(cone::EpiPerExp)
#     if !cone.inv_hess_updated
#         update_inv_hess(cone)
#     end
#     return
# end
#
# function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerExp)
#     update_inv_hess_prod(cone)
#     mul!(prod, cone.inv_hess, arr)
#     return prod
# end





# see analysis in https://github.com/lkapelevich/HypatiaBenchmarks.jl/tree/master/centralpoints
function get_central_ray_hypoperlog(w_dim::Int)
    if w_dim <= 10
        # lookup points where x = f'(x)
        return central_rays_hypoperlog[w_dim, :]
    end
    # use nonlinear fit for higher dimensions
    x = inv(w_dim)
    if w_dim <= 70
        u = -1.974777 * x ^ 2 + 0.413520 * x - 0.706751
        v = -1.213389 * x + 1.413551
        w = -0.406380 * x + 1.411894
    else
        u = 0.405290 * x - 0.707011
        v = -1.238597 * x + 1.414216
        w = -0.511055 * x + 1.414163
    end
    return [u, v, w]
end

const central_rays_hypoperlog = [
    -0.827838399  0.805102005  1.290927713;
    -0.751337431  0.980713381  1.317894791;
    -0.716423551  1.079796942  1.331762729;
    -0.699644766  1.144036715 1.341797042;
    -0.69134357  1.188706149  1.349742329;
    -0.687251501  1.221310686  1.3562255;
    -0.685353717  1.246016352  1.361602711;
    -0.684641818  1.265307905  1.366119586;
    -0.684585293  1.280747581  1.369956554;
    -0.684893372  1.293360445  1.373249434;
    ]
