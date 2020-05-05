#

mutable struct NewtonCache
    barrier::Function
    z::Vector{Float64}
    x::Vector{Float64}
    grad::Vector{Float64}
    hess::Matrix{Float64}
    stepdir::Vector{Float64}
    gradnorm::Float64

    function NewtonCache(barrier, z)
        n = 3
        nc = new()
        nc.barrier = barrier
        nc.z = z
        nc.x = zeros(n)
        nc.grad = zeros(n)
        nc.stepdir = zeros(n)
        nc.hess = zeros(n, n)
        return nc
    end
end

function step!(nc::NewtonCache)
    nc.grad .= ForwardDiff.gradient(nc.barrier, nc.x) + nc.z
    nc.hess .= ForwardDiff.hessian(nc.barrier, nc.x)
    nc.stepdir .= -Symmetric(nc.hess) \ nc.grad
    nc.gradnorm = -dot(nc.grad, nc.stepdir)
    return nc
end

function damped_newton_method(nc::NewtonCache)
    max_iter = 50
    iter = 0
    eta = 1e-14
    step!(nc)
    while nc.gradnorm > eta
        @. nc.x += 1 / (1 + nc.gradnorm) * nc.stepdir
        step!(nc)
        iter += 1
        iter > max_iter && error("iteration limit in Newton method")
    end
    return
end
