function conjugate_gradient(barrier::Function, check_feas::Function, z::Vector{T}) where {T}
    function modified_legendre(x)
        if !check_feas(x)
            return -Inf
        else
            return dot(z, x) + barrier(x)
        end
    end
    grad(x) = ForwardDiff.gradient(modified_legendre, x)
    hess(x) = ForwardDiff.hessian(modified_legendre, x)
    dfc = TwiceDifferentiableConstraints(fill(-T(Inf), size(z)), fill(T(Inf), size(z)))
    df = TwiceDifferentiable(modified_legendre, grad, hess, z, inplace = false)
    res = optimize(df, dfc, z, IPNewton())
    minimizer = Optim.minimizer(res)
    return -minimizer
end

barrier(x) = -sum(log.(x))
check_feas(x) = all(x .> zero(eltype(x)))
z = [1.0]

conjugate_gradient(barrier, check_feas, z)
