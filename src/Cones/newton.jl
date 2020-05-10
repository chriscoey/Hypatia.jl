


# for exp cone barrier
# TODO combine Hi * g into one
function newton_dir(u::T, v::T, w::T) where {T <: Real}
    lvw = log(w / v)
    vlwv = v * lvw
    vlwvu = vlwv - u
    denom = vlwvu + 2 * v
    wvdenom = u * v / denom
    vvdenom = (vlwvu + v) / denom

    g = zeros(T, 3)
    g[1] = inv(vlwvu)
    g[2] = (1 - lvw) / vlwvu - inv(v)
    g[3] = (-1 - v / vlwvu) / w

    Hi = zeros(T, 3, 3)
    Hi[3, 3] = w * vvdenom * w
    Hi[2, 2] = v * vvdenom * v
    Hi[1, 1] = 2 * (abs2(vlwv - v) + vlwv * (v - w)) + abs2(u) - v / denom * abs2(vluv - 2 * v)
    Hi[2, 3] = wvdenom * v
    Hi[1, 3] = wvdenom * (2 * vluv - u)
    Hi[1, 2] = (abs2(vluv) + u * (v - vluv)) / denom * v

    newton_dir = -(Symmetric(Hi, :U) * g)

    return newton_dir
end


function newton_step(cone::Cone)
    mock_cone = cone.newton_cone
    reset_data(mock_cone)
    load_point(mock_cone, cone.newton_point)
    @assert update_feas(mock_cone)
    g = grad(mock_cone)
    @. cone.newton_grad = -g - cone.dual_point
    inv_hess_prod!(cone.newton_stepdir, cone.newton_grad, mock_cone)
    cone.newton_norm = dot(cone.newton_grad, cone.newton_stepdir)
    return
end

function update_dual_grad(cone::Cone{T}) where {T <: Real}
    @assert cone.is_feas

    max_iter = 200 # TODO reduce: shouldn't really take > 40
    eta = eps(T) / 10 # TODO adjust
    # initial iterate
    copyto!(cone.newton_point, cone.point)
    newton_step(cone)
    # damped Newton
    iter = 0
    while cone.newton_norm > eta
        @. cone.newton_point += cone.newton_stepdir / (1 + cone.newton_norm)
        newton_step(cone)
        iter += 1
        # iter > max_iter && @warn("iteration limit in Newton method")
    end

    # can avoid a field unless we want to use switched Newton later
    @. cone.dual_grad = -cone.newton_point
    cone.dual_grad_updated = true

    # TODO remove check
    if norm(ForwardDiff.gradient(cone.barrier, cone.newton_point) + cone.dual_point) > sqrt(eps(T))
        @warn("conjugate grad calculation inaccurate")
    end

    return cone.dual_grad
end
