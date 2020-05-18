
using ForwardDiff
using DoubleFloats
using Quadmath

# for exp cone barrier
# TODO combine Hi * g into one
function hess_inv_dual_point(point::Vector{T}, dual_point::Vector{T}) where {T <: Real}
    # TODO toggle
    # newT = T
    # newT = Double64
    newT = Float128 # more precise than Double64
    # newT = BigFloat

    point = newT.(point)
    (u, v, w) = point
    @assert v > 0
    @assert w > 0
    lwv = log(w / v)
    vlwv = v * lwv
    vlwvu = vlwv - u
    @assert vlwvu > 0
    denom = vlwvu + 2 * v # v * (log(w / v) + 2) - u
    wvdenom = w * v / denom
    vvdenom = (vlwvu + v) / denom

    Hi = zeros(newT, 3, 3)
    Hi[1, 1] = 2 * (abs2(vlwv - v) + vlwv * (v - u)) + abs2(u) - v / denom * abs2(vlwv - 2 * v)
    Hi[1, 2] = (abs2(vlwv) + u * (v - vlwv)) / denom * v
    Hi[1, 3] = wvdenom * (vlwv + vlwvu)
    Hi[2, 2] = v * vvdenom * v
    Hi[2, 3] = wvdenom * v
    Hi[3, 3] = w * vvdenom * w
    Hi = Symmetric(Hi, :U)

    # if eigmin(Hi) < 100eps(T)
    #     @show Hi
    #     @show u, v, vlwvu
    # end

    Hiz = Hi * dual_point

    # hess_inv_dual_point = point - Hiz
    # nnorm = 3 + dot(dual_point, Hiz) - 2 * dot(point, dual_point)
    # if nnorm < -sqrt(eps(T))
    #     println()
    #     @show point
    #     @show dual_point
    #     @show nnorm
    #     @show Hi
    # end

    return Hiz
end

# function update_dual_grad_bf(cone::Cone{T}, mu::Real) where {T <: Real}
#     @assert cone.is_feas
#     point = cone.point
#     dual_point = BigFloat.(cone.dual_point)
#     curr = BigFloat.(cone.dual_grad)
#
#     max_iter = 200 # TODO reduce: shouldn't really take > 40
#     eta = sqrt(eps(T)) # TODO adjust
#
#     # damped Newton
#     curr .= point / mu
#     iter = 0
#     while true
#         (dir, Hiz) = hess_inv_dual_point(curr, dual_point)
#         nnorm = get_nu(cone) + dot(dual_point, Hiz) - 2 * dot(curr, dual_point)
#         denom = 1 + nnorm
#         @. curr += dir / denom
#         iter += 1
#         if nnorm < eta || iter >= max_iter
#             break
#         end
#     end
#
#     # # TODO remove check
#     # if norm(ForwardDiff.gradient(cone.barrier, curr) + cone.dual_point) > sqrt(eps(T))
#     #     @warn("conjugate grad calculation inaccurate")
#     # end
#
#     curr .*= -1
#     cone.dual_grad = Float64.(curr)
#
#     cone.dual_grad_updated = true
#     return cone.dual_grad
# end

function update_dual_grad(cone::Cone{T}, mu::T) where {T <: Real}
    @assert cone.is_feas
    @assert !cone.dual_grad_updated

    cone.dual_grad_inacc = false
    # bf_sol = update_dual_grad_bf(cone, mu)

    point = cone.point
    dual_point = cone.dual_point
    curr = cone.dual_grad

    # norm_z = norm(dual_point)
    # if norm_z > 1e5 || norm_z < 1e-5
    #     @warn("norm of dual point is $norm_z, maybe try scaling it in update_dual_grad")
    # end
    # inv_norm_z = inv(norm_z)
    # dual_point_scal = inv_norm_z * dual_point
    dual_point_scal = dual_point

    nu = get_nu(cone)
    max_iter = 25 # TODO make it depend on sqrt(nu). reduce: shouldn't really take > 40
    # max_iter = 50 # TODO useful for bigfloat
    eta = sqrt(eps(T)) # TODO adjust
    # neg_tol = eps(T)
    neg_tol = eta

    # initial guess based on central path proximity is point / mu
    # TODO depends on neighborhood definition we use, since that determines guarantees about relationship between s and conj_g
    # TODO skajaa ye central path is s + mu * conj_g = 0, so conj_g = -s/mu
    # TODO mosek central path is cone_nu = mu * dot(g, conj_g), satisfied by conj_g = -s/mu
    pscal = inv(mu)
    # TODO but point / cone_mu also works well (better actually) on the one case i tried (let cone_mu = dot(point, dual_point_scal) / cone_nu)
    # pscal = nu / dot(point, dual_point_scal)
    # @show dot(point, dual_point_scal) / nu / mu # TODO seems always close to 1
    curr .= pscal * point
    # curr .= point

    iter = 0
    while true
        iter += 1


        # # no scaling
        # # TODO to use, turn off dual_point scaling too
        # Hiz = hess_inv_dual_point(curr, dual_point_scal) # TODO just inv hess prod applied to dual_point
        # dir = curr - Hiz
        # nnorm = nu - dot(curr + dir, dual_point_scal)
        #
        # if nnorm < -neg_tol # bad nnorm
        #     @show nnorm, iter
        #     # cone.dual_grad_inacc = true
        #     break
        # # elseif nnorm < 0 # bad nnorm
        # #     nnorm *= 10
        # end
        #
        # # damped Newton step
        # alpha = inv(1 + abs(nnorm))
        # axpy!(alpha, dir, curr)


        # scaling
        scalval = norm(curr)
        inv_scalval = inv(scalval)
        # (u, v, w) = curr
        # inv_scalval = v * (log(w / v) + 2) - u
        # scalval = inv(inv_scalval)
        curr_scal = inv_scalval * curr
        Hiz_scal = hess_inv_dual_point(curr_scal, dual_point_scal) # TODO just inv hess prod applied to dual_point
        dir_scal = curr_scal - scalval * Hiz_scal
        nnorm = nu - scalval * dot(curr_scal + dir_scal, dual_point_scal)

        if nnorm < -neg_tol # bad nnorm
            # @show nnorm, iter
            # @show norm(dual_point)
            cone.dual_grad_inacc = true
            break
        elseif nnorm < 0 # bad nnorm
            nnorm *= 10
        end

        # if nnorm > 0.35
            # damped Newton step
            alpha = scalval / (1 + abs(nnorm))
        # else
        #     alpha = scalval
        # end
        axpy!(alpha, dir_scal, curr)

        if nnorm < eta
            break
        elseif iter >= max_iter || dot(curr, dual_point) < 0
            @show nnorm, iter
            cone.dual_grad_inacc = true
            break
        end
    end

    curr .*= -1
    # curr .*= -inv_norm_z

    cgnorm = norm(ForwardDiff.gradient(cone.barrier, -curr) + cone.dual_point)
    if cgnorm > 1000sqrt(eps(T))
        println("conjugate grad calculation inaccurate: $cgnorm")
    end

    cone.dual_grad_updated = true
    return cone.dual_grad
end


# function newton_step(cone::Cone)
#     mock_cone = cone.newton_cone
#     reset_data(mock_cone)
#     load_point(mock_cone, cone.newton_point)
#     @assert update_feas(mock_cone)
#     g = grad(mock_cone)
#     @. cone.newton_grad = -g - cone.dual_point
#     inv_hess_prod!(cone.newton_stepdir, cone.newton_grad, mock_cone)
#     cone.newton_norm = dot(cone.newton_grad, cone.newton_stepdir)
#     return
# end

# function update_dual_grad(cone::Cone{T}) where {T <: Real}
#     @assert cone.is_feas
#
#     max_iter = 200 # TODO reduce: shouldn't really take > 40
#     eta = eps(T) / 10 # TODO adjust
#     # initial iterate
#     copyto!(cone.newton_point, cone.point)
#     newton_step(cone)
#     # damped Newton
#     iter = 0
#     while cone.newton_norm > eta
#         @. cone.newton_point += cone.newton_stepdir / (1 + cone.newton_norm)
#         newton_step(cone)
#         iter += 1
#         # iter > max_iter && @warn("iteration limit in Newton method")
#     end
#
#     # can avoid a field unless we want to use switched Newton later
#     @. cone.dual_grad = -cone.newton_point
#     cone.dual_grad_updated = true
#
#     # TODO remove check
#     if norm(ForwardDiff.gradient(cone.barrier, cone.newton_point) + cone.dual_point) > sqrt(eps(T))
#         @warn("conjugate grad calculation inaccurate")
#     end
#
#     return cone.dual_grad
# end
