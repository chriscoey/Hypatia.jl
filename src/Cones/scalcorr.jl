
# TODO later move to Cones.jl or elsewhere

import Optim
import ForwardDiff

use_scaling(cone::Cone) = false

use_correction(cone::Cone) = false

scal_hess(cone::Cone{T}, mu::T, z::AbstractVector{T}) where {T} = (cone.scal_hess_updated ? cone.scal_hess : update_scal_hess(cone, mu, z))

function update_scal_hess(
    cone::Cone{T},
    mu::T,
    z::AbstractVector{T}; # dual point
    use_update_1::Bool = true, # easy update
    use_update_2::Bool = true, # hard update
    ) where {T}
    @assert is_feas(cone)
    @assert !cone.scal_hess_updated
    s = cone.point
    # z = cone.dual_point

    scal_hess = mu * hess(cone)

    if use_update_1
        # first update
        denom_a = dot(s, z)
        muHs = scal_hess * s
        denom_b = dot(s, muHs)

        if denom_a > 0
            scal_hess += Symmetric(z * z') / denom_a
        end
        if denom_b > 0
            scal_hess -= Symmetric(muHs * muHs') / denom_b
        end

        @show norm(scal_hess * s - z)
    end

    if use_update_2
        # second update
        g = grad(cone)
        conj_g = conjugate_gradient(barrier(cone), s, z)

        mu_cone = dot(s, z) / get_nu(cone)
        # @show mu_cone
        dual_gap = z + mu_cone * g
        primal_gap = s + mu_cone * conj_g
        # dual_gap = z + mu * g
        # primal_gap = s + mu * conj_g

        denom_a = dot(primal_gap, dual_gap)
        H1prgap = scal_hess * primal_gap
        denom_b = dot(primal_gap, H1prgap)

        if denom_a > 0
            scal_hess += Symmetric(dual_gap * dual_gap') / denom_a
        end
        if denom_b > 0
            scal_hess -= Symmetric(H1prgap * H1prgap') / denom_b
        end

        # @show primal_gap, dual_gap
        @show norm(scal_hess * s - z)
        @show norm(scal_hess * -conj_g + g)
        @show norm(scal_hess * primal_gap - dual_gap)
    end

    copyto!(cone.scal_hess, scal_hess)

    cone.scal_hess_updated = true
    return cone.scal_hess
end

# TODO use domain constraints properly
# TODO use central point as starting point?
# TODO use constrained method from https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/ipnewton_basics/#constrained-optimization-with-ipnewton
function conjugate_gradient(barrier::Function, s::AbstractVector{T}, z::AbstractVector{T}) where {T}
    modified_legendre(x) = ((x[2] <= 0 || x[3] <= 0 || x[2] * log(x[3] / x[2]) - x[1] <= 0) ? 1e16 : dot(z, x) + barrier(x)) # TODO bad feas check - need constraints
    res = Optim.optimize(modified_legendre, s, Optim.Newton())
    minimizer = Optim.minimizer(res)
    @assert !any(isnan, minimizer)
    return -minimizer
end

# correction
function correction(cone::Cone{T}, primal_dir::AbstractVector{T}, dual_dir::AbstractVector{T}) where {T}
    dim = cone.dim
    point = cone.point
    FD_3deriv = ForwardDiff.jacobian(x -> ForwardDiff.hessian(barrier(cone), x), point)
    # check log-homog property that F'''(point)[point] = -2F''(point)
    @assert reshape(FD_3deriv * cone.point, dim, dim) ≈ -2 * ForwardDiff.hessian(barrier(cone), point)
    Hinv_z = inv_hess_prod!(similar(dual_dir), dual_dir, cone)
    FD_corr = reshape(FD_3deriv * primal_dir, dim, dim) * Hinv_z / -2
    return FD_corr
end
