#
import Hypatia # for grad/hess hacks

mutable struct NewtonCache{T <: Real}
    barrier::Function
    z::AbstractVector{T}
    val::T
    x::Vector{T}
    grad::Vector{T}
    hess::Matrix{T}
    stepdir::Vector{T}
    gradnorm::T

    function NewtonCache{T}(barrier, z::AbstractVector{T}) where {T <: Real}
        n = 3
        nc = new{T}()
        nc.barrier = barrier
        nc.z = z
        nc.x = zeros(T, n)
        nc.grad = zeros(T, n)
        nc.stepdir = zeros(T, n)
        nc.hess = zeros(T, n, n)
        return nc
    end
end

# hacks
function update_grad(nc::NewtonCache{T}) where {T <: Real}
    cone = Hypatia.Cones.HypoPerLog{T}(3)
    Hypatia.Cones.setup_data(cone)
    cone.point = nc.x
    Hypatia.Cones.update_feas(cone)
    Hypatia.Cones.update_grad(cone)
    nc.grad .= cone.grad
    return nc.grad
end
function update_hess(nc::NewtonCache{T}) where {T <: Real}
    cone = Hypatia.Cones.HypoPerLog{T}(3)
    Hypatia.Cones.setup_data(cone)
    cone.point = nc.x
    Hypatia.Cones.update_feas(cone)
    Hypatia.Cones.update_grad(cone)
    Hypatia.Cones.update_hess(cone)
    nc.hess .= cone.hess.data
    return nc.hess
end

function step!(nc::NewtonCache)
    # nc.grad .= ForwardDiff.gradient(nc.barrier, nc.x) + nc.z
    # nc.hess .= ForwardDiff.hessian(nc.barrier, nc.x)
    nc.grad .= update_grad(nc) + nc.z
    nc.hess .= update_hess(nc)
    nc.stepdir .= -Symmetric(nc.hess) \ nc.grad
    nc.gradnorm = -dot(nc.grad, nc.stepdir)
    return nc
end

# unused if not using switched_newton
function update_val!(nc::NewtonCache)
    x = nc.x
    nc.val = (x[2] <= 0 || x[3] <= 0 || x[2] * log(x[3] / x[2]) - x[1] <= 0) ? Inf : (nc.barrier(x) + dot(x, nc.z))
    return nc.val
end
function linesearch(nc::NewtonCache, alpha::Float64, beta::Float64)
    max_iter = 500
    t = 1.0
    anorm = alpha * nc.gradnorm
    x0 = copy(nc.x)
    iter = 0

    f_prev = nc.val
    @. nc.x = x0 + t * nc.stepdir
    while update_val!(nc) > f_prev + t * anorm
        t *= beta
        @. nc.x = x0 + t * nc.stepdir
        iter += 1
        iter > max_iter && error("iteration limit in linesearch")
    end
    return t
end

function switched_newton_method(nc::NewtonCache{T}) where {T <: Real}
    max_iter = 500
    iter = 0
    eta_1 = eps(T) * 1000 # 1e-14
    eta_2 = sqrt(eps(T)) / 1000
    step!(nc)
    # damped Newton
    while nc.gradnorm > eta_1
        @. nc.x += 1 / (1 + nc.gradnorm) * nc.stepdir
        step!(nc)
        iter += 1
        iter > max_iter && error("iteration limit in Newton method")
    end
    # @show norm(nc.grad)
    # switch to regular Newton with lineasearch
    # iter = 0
    # update_val!(nc)
    # while nc.gradnorm > eta_2
    #     t = linesearch(nc, 1e-6, 0.5)
    #     step!(nc)
    #     iter += 1
    #     iter > max_iter && error("iteration limit in Newton method")
    # end
    return
end
