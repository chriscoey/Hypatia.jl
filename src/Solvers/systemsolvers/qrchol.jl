#=
Copyright 2018, Chris Coey and contributors

QR+Cholesky linear system solver
requires precomputed QR factorization of A'

solves linear system in naive.jl by first eliminating s, kap, and tau via the method in the symindef solver, then reducing the 3x3 symmetric indefinite system to a series of low-dimensional operations via a procedure similar to that described by S10.3 of
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf (the dominating subroutine is a positive definite linear solve with RHS of dimension n-p x 3)
=#

abstract type QRCholSystemSolver{T <: Real} <: SystemSolver{T} end

function solve_system(system_solver::QRCholSystemSolver{T}, solver::Solver{T}, sol::Vector{T}, rhs::Vector{T}) where {T <: Real}
    model = solver.model
    x_rows = system_solver.x_rows
    y_rows = system_solver.y_rows
    z_rows = system_solver.z_rows

    rhs3 = system_solver.rhs3
    dim3 = length(rhs3)

    @. @views rhs3[x_rows] = rhs[x_rows]
    @. @views rhs3[y_rows] = -rhs[y_rows]

    for (cone_k, idxs_k) in zip(model.cones, model.cone_idxs)
        z_rows_k = (model.n + model.p) .+ idxs_k
        z_k = @view rhs[z_rows_k]
        z3_k = @view rhs3[z_rows_k]
        s_k = @view rhs[(dim3 + 1) .+ idxs_k]

        if Cones.use_dual_barrier(cone_k)
            z_temp_k = @view sol[z_rows_k]
            @. z_temp_k = -z_k - s_k
            Cones.inv_hess_prod!(z3_k, z_temp_k, cone_k)
            z3_k ./= solver.mu
        else
            # if Cones.use_scaling(cone_k)
                Cones.scal_hess_prod!(z3_k, z_k, cone_k, solver.mu)
                axpby!(-1, s_k, -1, z3_k)
            # else
            #     Cones.hess_prod!(z3_k, z_k, cone_k)
            #     axpby!(-1, s_k, -solver.mu, z3_k)
            # end
        end
    end

    sol3 = solve_subsystem(system_solver, solver, rhs3) # NOTE modifies and returns rhs3

    # TODO refactor all below
    # TODO maybe use higher precision here
    const_sol = system_solver.const_sol

    kapontau = solver.kap / solver.tau

    # lift to get tau
    @views tau_num = rhs[dim3 + 1] + rhs[end] + dot(model.c, sol3[x_rows]) + dot(model.b, sol3[y_rows]) + dot(model.h, sol3[z_rows])
    @views tau_denom = kapontau - dot(model.c, const_sol[x_rows]) - dot(model.b, const_sol[y_rows]) - dot(model.h, const_sol[z_rows])

    sol_tau = tau_num / tau_denom
    @. sol[1:dim3] = sol3 + sol_tau * const_sol
    sol[dim3 + 1] = sol_tau

    # lift to get s and kap
    # TODO refactor below for use with symindef and qrchol methods
    s = @view sol[(dim3 + 2):(end - 1)]
    @. @views s = model.h * sol_tau - rhs[z_rows]
    @views mul!(s, model.G, sol[x_rows], -1, true)

    # NT: kap = -kapbar/taubar*tau + kaprhs
    sol[end] = -kapontau * sol_tau + rhs[end]

    return sol
end

function solve_subsystem(system_solver::QRCholSystemSolver{T}, solver::Solver{T}, rhs3::Vector{T}) where {T <: Real}
    model = solver.model
    (n, p) = (model.n, model.p)
    @views x = rhs3[system_solver.x_rows]
    @views y = rhs3[system_solver.y_rows]
    @views z = rhs3[system_solver.z_rows]

    copyto!(system_solver.QpbxGHbz, x) # TODO can be avoided
    mul!(system_solver.QpbxGHbz, model.G', z, true, true)
    lmul!(solver.Ap_Q', system_solver.QpbxGHbz)

    if !iszero(p)
        ldiv!(solver.Ap_R', y)
        rhs3[1:p] = y

        if !isempty(system_solver.Q2div)
            mul!(system_solver.GQ1x, system_solver.GQ1, y)
            block_hess_prod.(model.cones, system_solver.HGQ1x_k, system_solver.GQ1x_k, solver.mu)
            mul!(system_solver.Q2div, system_solver.GQ2', system_solver.HGQ1x, -1, true)
        end
    end

    if !isempty(system_solver.Q2div)
        @views x_sub2 = rhs3[(p + 1):n]
        # @show norm(system_solver.Q2div)
        mul!(x_sub2, solver.mu, system_solver.Q2div)
        # @show norm(x_sub2)
        inv_prod(system_solver.fact_cache, x_sub2)
    end

    lmul!(solver.Ap_Q, x)

    mul!(system_solver.Gx, model.G, x)
    block_hess_prod.(model.cones, system_solver.HGx_k, system_solver.Gx_k, solver.mu)

    @. z = system_solver.HGx - z

    if !iszero(p)
        copyto!(y, system_solver.Q1pbxGHbz)
        mul!(y, system_solver.GQ1', system_solver.HGx, -1, true)
        ldiv!(solver.Ap_R, y)
    end

    return rhs3
end

function block_hess_prod(cone_k::Cones.Cone{T}, prod_k::AbstractVecOrMat{T}, arr_k::AbstractVecOrMat{T}, mu::T) where {T <: Real}
    if Cones.use_dual_barrier(cone_k)
        Cones.inv_hess_prod!(prod_k, arr_k, cone_k)
        @. prod_k /= mu
    else
        # if Cones.use_scaling(cone_k)
            Cones.scal_hess_prod!(prod_k, arr_k, cone_k, mu)
        # else
        #     Cones.hess_prod!(prod_k, arr_k, cone_k)
        #     @. prod_k *= mu
        # end
    end
    return
end

#=
direct dense
=#

mutable struct QRCholDenseSystemSolver{T <: Real} <: QRCholSystemSolver{T}
    x_rows::UnitRange{Int}
    y_rows::UnitRange{Int}
    z_rows::UnitRange{Int}
    rhs3::Vector{T}
    const_sol::Vector{T}
    lhs1::Symmetric{T, Matrix{T}}
    inv_hess_cones::Vector{Int}
    inv_hess_sqrt_cones::Vector{Int}
    hess_cones::Vector{Int}
    hess_sqrt_cones::Vector{Int}
    GQ1
    GQ2
    QpbxGHbz
    Q1pbxGHbz
    Q2div
    GQ1x
    HGQ1x
    HGQ2
    Gx
    HGx
    HGQ1x_k
    GQ1x_k
    HGQ2_k
    GQ2_k
    HGx_k
    Gx_k
    fact_cache::Union{DensePosDefCache{T}, DenseSymCache{T}} # can use BunchKaufman or Cholesky
    function QRCholDenseSystemSolver{T}(;
        fact_cache::Union{DensePosDefCache{T}, DenseSymCache{T}} = DensePosDefCache{T}(), # NOTE or DenseSymCache{T}()
        # fact_cache::Union{DensePosDefCache{T}, DenseSymCache{T}} = DenseSymCache{T}(), # NOTE or DenseSymCache{T}()
        ) where {T <: Real}
        system_solver = new{T}()
        system_solver.fact_cache = fact_cache
        return system_solver
    end
end

function load(system_solver::QRCholDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    nmp = n - p
    cone_idxs = model.cone_idxs

    system_solver.x_rows = 1:n
    system_solver.y_rows = n .+ (1:p)
    system_solver.z_rows = (n + p) .+ (1:q)

    system_solver.rhs3 = Vector{T}(undef, n + p + q)
    system_solver.lhs1 = Symmetric(Matrix{T}(undef, nmp, nmp), :U)

    num_cones = length(cone_idxs)
    system_solver.inv_hess_cones = sizehint!(Int[], num_cones)
    system_solver.inv_hess_sqrt_cones = sizehint!(Int[], num_cones)
    system_solver.hess_cones = sizehint!(Int[], num_cones)
    system_solver.hess_sqrt_cones = sizehint!(Int[], num_cones)

    # NOTE very inefficient method used for sparse G * QRSparseQ : see https://github.com/JuliaLang/julia/issues/31124#issuecomment-501540818
    @timeit solver.timer "mul_G_Q" GQ = model.G * solver.Ap_Q

    system_solver.GQ2 = GQ[:, (p + 1):end]
    system_solver.HGQ2 = zeros(T, q, nmp)
    system_solver.QpbxGHbz = Vector{T}(undef, n)
    system_solver.Q2div = view(system_solver.QpbxGHbz, (p + 1):n)
    system_solver.Gx = Vector{T}(undef, q)
    system_solver.HGx = similar(system_solver.Gx)
    system_solver.HGQ2_k = [view(system_solver.HGQ2, idxs, :) for idxs in cone_idxs]
    system_solver.GQ2_k = [view(system_solver.GQ2, idxs, :) for idxs in cone_idxs]
    system_solver.HGx_k = [view(system_solver.HGx, idxs, :) for idxs in cone_idxs]
    system_solver.Gx_k = [view(system_solver.Gx, idxs, :) for idxs in cone_idxs]

    if !iszero(p)
        system_solver.GQ1 = GQ[:, 1:p]
        system_solver.Q1pbxGHbz = view(system_solver.QpbxGHbz, 1:p)
        system_solver.GQ1x = Vector{T}(undef, q)
        system_solver.HGQ1x = similar(system_solver.GQ1x)
        system_solver.HGQ1x_k = [view(system_solver.HGQ1x, idxs, :) for idxs in cone_idxs]
        system_solver.GQ1x_k = [view(system_solver.GQ1x, idxs, :) for idxs in cone_idxs]
    end

    load_matrix(system_solver.fact_cache, system_solver.lhs1)

    system_solver.const_sol = similar(system_solver.rhs3)

    return system_solver
end

# NOTE move to dense.jl if useful elsewhere
outer_prod(A::AbstractMatrix{T}, B::AbstractMatrix{T}, alpha::Real, beta::Real) where {T <: LinearAlgebra.BlasReal} = BLAS.syrk!('U', 'T', alpha, A, beta, B)
outer_prod(A::AbstractMatrix{T}, B::AbstractMatrix{T}, alpha::Real, beta::Real) where {T <: Real} = mul!(B, A', A, alpha, beta)

function update_lhs(system_solver::QRCholDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    lhs = system_solver.lhs1.data

    if !isempty(system_solver.Q2div)
        inv_hess_cones = empty!(system_solver.inv_hess_cones)
        inv_hess_sqrt_cones = empty!(system_solver.inv_hess_sqrt_cones)
        hess_cones = empty!(system_solver.hess_cones)
        hess_sqrt_cones = empty!(system_solver.hess_sqrt_cones)

        # update hessian factorizations and partition of cones
        for (k, cone_k) in enumerate(model.cones)
            if hasfield(typeof(cone_k), :hess_fact_cache) # TODO use dispatch or a function
                @assert !cone_k.scal_hess_updated
                Cones.update_scal_hess(cone_k, solver.mu)
                fact_success = Cones.update_hess_fact(cone_k, recover = false)
                # if cone_k.hess_fact_cache isa DenseSymCache{T}
                #     cones_list = Cones.use_dual_barrier(cone_k) ? inv_hess_cones : hess_cones
                #     push!(cones_list, k)
                #     continue
                # end
                cones_list = (fact_success ? hess_sqrt_cones : hess_cones)
                push!(cones_list, k)
            else
                cones_list = Cones.use_dual_barrier(cone_k) ? inv_hess_sqrt_cones : hess_sqrt_cones
                push!(cones_list, k)
            end
        end

        # TODO inv cones
        @assert isempty(inv_hess_cones)
        @assert isempty(inv_hess_sqrt_cones)
        # lhs .= 0
        # # do inv_hess and inv_hess_sqrt cones
        # if isempty(inv_hess_sqrt_cones)
        #     lhs .= 0
        # else
        #     idx = 1
        #     for k in inv_hess_sqrt_cones
        #         arr_k = system_solver.GQ2_k[k]
        #         q_k = size(arr_k, 1)
        #         @views prod_k = system_solver.HGQ2[idx:(idx + q_k - 1), :]
        #         Cones.inv_hess_sqrt_prod!(prod_k, arr_k, model.cones[k])
        #         idx += q_k
        #     end
        #     @views HGQ2_sub = system_solver.HGQ2[1:(idx - 1), :]
        #     outer_prod(HGQ2_sub, lhs, true, false)
        # end
        #
        # for k in inv_hess_cones
        #     arr_k = system_solver.GQ2_k[k]
        #     prod_k = system_solver.HGQ2_k[k]
        #     Cones.inv_hess_prod!(prod_k, arr_k, model.cones[k])
        #     mul!(lhs, arr_k', prod_k, true, true)
        # end
        #
        # if !(isempty(inv_hess_cones) && isempty(inv_hess_sqrt_cones))
        #     # divide by mu for inv_hess and inv_hess_sqrt cones
        #     lhs ./= solver.mu
        # end

        # do hess and hess_sqrt cones
        # @assert isempty(hess_cones)
        if !isempty(hess_sqrt_cones)
            idx = 1
            for k in hess_sqrt_cones
                arr_k = system_solver.GQ2_k[k]
                q_k = size(arr_k, 1)
                @views prod_k = system_solver.HGQ2[idx:(idx + q_k - 1), :]
                # Cones.hess_sqrt_prod!(prod_k, arr_k, model.cones[k])
                Cones.scal_hess_sqrt_prod!(prod_k, arr_k, model.cones[k], solver.mu)
                idx += q_k
            end
            @views HGQ2_sub = system_solver.HGQ2[1:(idx - 1), :]
            # outer_prod(HGQ2_sub, lhs, solver.mu, true)
            outer_prod(HGQ2_sub, lhs, true, false)
        end

        for k in hess_cones
            arr_k = system_solver.GQ2_k[k]
            prod_k = system_solver.HGQ2_k[k]
            # Cones.hess_prod!(prod_k, arr_k, model.cones[k])
            Cones.scal_hess_prod!(prod_k, arr_k, model.cones[k], solver.mu)
            mul!(lhs, arr_k', prod_k, true, true)
            # mul!(lhs, arr_k', prod_k, solver.mu, true)
        end

        # # TODO only do scal hess prod:
        # lhs .= 0
        # for k in eachindex(model.cones)
        #     arr_k = system_solver.GQ2_k[k]
        #     prod_k = system_solver.HGQ2_k[k]
        #     cone_k = model.cones[k]
        #     # if Cones.use_scaling(cone_k)
        #         Cones.scal_hess_prod!(prod_k, arr_k, cone_k, solver.mu)
        #         mul!(lhs, arr_k', prod_k, true, true)
        #     # else
        #     #     Cones.hess_prod!(prod_k, arr_k, cone_k)
        #     #     mul!(lhs, arr_k', prod_k, solver.mu, true)
        #     # end
        # end
    end

    # println()
    # # @show solver.mu
    # @show norm(system_solver.lhs1)
    # @show extrema(eigvals(system_solver.lhs1))
    lmul!(solver.mu, system_solver.lhs1.data)
    # # system_solver.lhs1 += sqrt(eps(T)) * I # attempt recovery # TODO make more efficient
    # @show norm(system_solver.lhs1)
    # @show extrema(eigvals(system_solver.lhs1))

    # TODO refactor below
    # TODO if cholesky fails, add to diagonal (maybe based on norm of diagonal elements)
    if !isempty(system_solver.lhs1) && !update_fact(system_solver.fact_cache, system_solver.lhs1)
        @warn("QRChol factorization failed")
        if T <: LinearAlgebra.BlasReal && system_solver.fact_cache isa DensePosDefCache{T}
            @warn("switching QRChol solver from Cholesky to Bunch Kaufman")
            system_solver.fact_cache = DenseSymCache{T}()
            load_matrix(system_solver.fact_cache, system_solver.lhs1)
        else
            system_solver.lhs1 += sqrt(eps(T)) * I # attempt recovery # TODO make more efficient
        end
        if !update_fact(system_solver.fact_cache, system_solver.lhs1)
            system_solver.lhs1 += sqrt(eps(T)) * I # attempt recovery # TODO make more efficient
            update_fact(system_solver.fact_cache, system_solver.lhs1) || @warn("QRChol Bunch-Kaufman factorization failed after recovery")
        end
    end

    # update solution for fixed c,b,h part
    const_sol = system_solver.const_sol
    @. const_sol[system_solver.x_rows] = -model.c
    const_sol[system_solver.y_rows] = model.b
    @views const_sol_z = const_sol[system_solver.z_rows]
    for (cone_k, idxs_k) in zip(model.cones, model.cone_idxs)
        @views block_hess_prod(cone_k, const_sol_z[idxs_k], model.h[idxs_k], solver.mu)
    end
    solve_subsystem(system_solver, solver, const_sol)

    return system_solver
end
