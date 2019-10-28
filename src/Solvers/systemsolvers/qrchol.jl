#=
Copyright 2018, Chris Coey and contributors

QR+Cholesky linear system solver
requires precomputed QR factorization of A'
solves linear system in naive.jl by first eliminating s, kap, and tau via the method in the symindef solver, then reducing the 3x3 symmetric indefinite system to a series of low-dimensional operations via a procedure similar to that described by S10.3 of
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf (the dominating subroutine is a positive definite linear solve with side dimension n-p)
=#

abstract type QRCholSystemSolver{T <: Real} <: SystemSolver{T} end

function solve_system(system_solver::QRCholSystemSolver{T}, solver::Solver{T}, sol::Vector{T}, rhs::Vector{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)

    rhs3 = system_solver.rhs3
    dim3 = length(rhs3)
    x_rows = 1:n
    y_rows = n .+ (1:p)
    z_rows = (n + p) .+ (1:q)

    @. @views rhs3[x_rows] = rhs[x_rows]
    @. @views rhs3[y_rows] = -rhs[y_rows]

    for (cone_k, idxs_k) in zip(model.cones, model.cone_idxs)
        z_rows_k = (n + p) .+ idxs_k
        z_k = @view rhs[z_rows_k]
        z3_k = @view rhs3[z_rows_k]
        s_rows_k = (dim3 + 1) .+ idxs_k
        s_k = @view rhs[s_rows_k]

        if Cones.use_dual(cone_k) # no scaling
            z_temp_k = @view sol[z_rows_k]
            @. z_temp_k = -z_k - s_k
            Cones.inv_hess_prod!(z3_k, z_temp_k, cone_k)
            z3_k ./= solver.mu
        else
            Cones.hess_prod!(z3_k, z_k, cone_k)
            if Cones.use_scaling(cone_k)
                axpby!(-1, s_k, -1, z3_k)
            else
                axpby!(-1, s_k, -solver.mu, z3_k)
            end
        end
    end

    sol3 = solve_3x3_subsystem(system_solver, solver, rhs3) # NOTE modifies and returns rhs3

    # TODO refactor all below
    # TODO maybe use higher precision here
    kapontau = solver.kap / solver.tau
    const_sol = system_solver.const_sol

    # lift to get tau
    @views tau_num = rhs[dim3 + 1] + kapontau * rhs[end] + dot(model.c, sol3[x_rows]) + dot(model.b, sol3[y_rows]) + dot(model.h, sol3[z_rows])
    @views tau_denom = kapontau - dot(model.c, const_sol[x_rows]) - dot(model.b, const_sol[y_rows]) - dot(model.h, const_sol[z_rows])
    sol_tau = tau_num / tau_denom

    @. sol[1:dim3] = sol3 + sol_tau * const_sol
    sol[dim3 + 1] = sol_tau

    # lift to get s and kap
    # TODO refactor below for use with symindef and qrchol methods
    # s = -G*x + h*tau - zrhs
    s = @view sol[(dim3 + 2):(end - 1)]
    @. @views s = model.h * sol_tau - rhs[z_rows]
    @views mul!(s, model.G, sol[x_rows], -1, true)

    # kap = kapbar/taubar*(kaprhs - tau)
    sol[end] = kapontau * (rhs[end] - sol_tau)

    return sol
end

function solve_3x3_subsystem(system_solver::QRCholSystemSolver{T}, solver::Solver{T}, rhs3::Vector{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)

    x = @view rhs3[1:n]
    x_sub1 = @view rhs3[1:p]
    x_sub2 = @view rhs3[(p + 1):n]
    y = @view rhs3[n .+ (1:p)]
    z = @view rhs3[(n + p) .+ (1:q)]

    ldiv!(solver.Ap_R', y)

    copyto!(system_solver.QpbxGHbz, x) # TODO can be avoided
    mul!(system_solver.QpbxGHbz, model.G', z, true, true)
    lmul!(solver.Ap_Q', system_solver.QpbxGHbz)

    copyto!(x_sub1, y)

    if !isempty(system_solver.Q2div)
        mul!(system_solver.GQ1x, system_solver.GQ1, y)
        block_hessian_product(solver, model.cones, system_solver.HGQ1x_k, system_solver.GQ1x_k)
        mul!(system_solver.Q2div, system_solver.GQ2', system_solver.HGQ1x, -1, true)

        solve_subsystem(system_solver, x_sub2, system_solver.Q2div)
    end

    lmul!(solver.Ap_Q, x)

    mul!(system_solver.Gx, model.G, x)
    block_hessian_product(solver, model.cones, system_solver.HGx_k, system_solver.Gx_k)

    @. z = system_solver.HGx - z

    if !isempty(y)
        copyto!(y, system_solver.Q1pbxGHbz)
        mul!(y, system_solver.GQ1', system_solver.HGx, -1, true)
        ldiv!(solver.Ap_R, y)
    end

    return rhs3
end

function block_hessian_product(solver::Solver, cones::Vector, prods::Vector, arrs::Vector)
    for (cone_k, prod_k, arr_k) in zip(cones, prods, arrs)
        if Cones.use_dual(cone_k) # no scaling
            Cones.inv_hess_prod!(prod_k, arr_k, cone_k)
            prod_k ./= solver.mu
        else
            Cones.hess_prod!(prod_k, arr_k, cone_k)
            if !Cones.use_scaling(cone_k)
                prod_k .*= solver.mu
            end
        end
    end
end

#=
direct dense
=#

mutable struct QRCholDenseSystemSolver{T <: Real} <: QRCholSystemSolver{T}
    rhs3::Vector{T}
    const_sol::Vector{T}
    lhs1::Symmetric{T, Matrix{T}}
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
        fact_cache::Union{DensePosDefCache{T}, DenseSymCache{T}} = (T <: LinearAlgebra.BlasReal ? DenseSymCache{T}() : DensePosDefCache{T}()),
        ) where {T <: Real}
        system_solver = new{T}()
        system_solver.fact_cache = fact_cache
        return system_solver
    end
end

function load(system_solver::QRCholDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cone_idxs = model.cone_idxs

    system_solver.rhs3 = zeros(T, n + p + q)

    # TODO optimize for case of empty A
    # TODO very inefficient method used for sparse G * QRSparseQ : see https://github.com/JuliaLang/julia/issues/31124#issuecomment-501540818
    if !isa(model.G, Matrix{T})
        @warn("in QRChol, converting G to dense before multiplying by sparse Householder Q due to very inefficient dispatch")
    end
    G = Matrix(model.G)
    GQ = rmul!(G, solver.Ap_Q)

    system_solver.GQ1 = GQ[:, 1:p]
    system_solver.GQ2 = GQ[:, (p + 1):end]
    nmp = n - p
    system_solver.HGQ2 = Matrix{T}(undef, q, nmp)
    system_solver.lhs1 = Symmetric(Matrix{T}(undef, nmp, nmp), :U)
    system_solver.QpbxGHbz = Vector{T}(undef, n)
    system_solver.Q1pbxGHbz = view(system_solver.QpbxGHbz, 1:p)
    system_solver.Q2div = view(system_solver.QpbxGHbz, (p + 1):n)
    system_solver.GQ1x = Vector{T}(undef, q)
    system_solver.HGQ1x = similar(system_solver.GQ1x)
    system_solver.Gx = similar(system_solver.GQ1x)
    system_solver.HGx = similar(system_solver.Gx)
    system_solver.HGQ1x_k = [view(system_solver.HGQ1x, idxs, :) for idxs in cone_idxs]
    system_solver.GQ1x_k = [view(system_solver.GQ1x, idxs, :) for idxs in cone_idxs]
    system_solver.HGQ2_k = [view(system_solver.HGQ2, idxs, :) for idxs in cone_idxs]
    system_solver.GQ2_k = [view(system_solver.GQ2, idxs, :) for idxs in cone_idxs]
    system_solver.HGx_k = [view(system_solver.HGx, idxs, :) for idxs in cone_idxs]
    system_solver.Gx_k = [view(system_solver.Gx, idxs, :) for idxs in cone_idxs]

    load_matrix(system_solver.fact_cache, system_solver.lhs1, copy_A = false) # overwrite lhs1 with new factorization each time update_fact is called

    system_solver.const_sol = similar(system_solver.rhs3)

    return system_solver
end

function update_fact(system_solver::QRCholDenseSystemSolver, solver::Solver)
    isempty(system_solver.Q2div) && return system_solver
    model = solver.model

    # TODO can be faster and numerically better for some cones to use L factor of cholesky of hessian here, with BLAS.syrk updating
    block_hessian_product(solver, model.cones, system_solver.HGQ2_k, system_solver.GQ2_k)
    mul!(system_solver.lhs1.data, system_solver.GQ2', system_solver.HGQ2)

    update_fact(system_solver.fact_cache, system_solver.lhs1) # overwrites lhs1 with new factorization

    (n, p) = (model.n, model.p)
    const_sol = system_solver.const_sol
    @views const_sol[1:n] = -model.c
    @views const_sol[n .+ (1:p)] = model.b
    for (cone_k, idxs_k) in zip(model.cones, model.cone_idxs)
        h_k = @view model.h[idxs_k]
        z3_k = @view const_sol[(n + p) .+ idxs_k]

        if Cones.use_dual(cone_k) # no scaling
            Cones.inv_hess_prod!(z3_k, h_k, cone_k)
            z3_k ./= solver.mu
        else
            Cones.hess_prod!(z3_k, h_k, cone_k)
            if !Cones.use_scaling(cone_k)
                z3_k .*= solver.mu
            end
        end
    end
    solve_3x3_subsystem(system_solver, solver, const_sol)

    return system_solver
end

function solve_subsystem(system_solver::QRCholDenseSystemSolver, sol1::AbstractVector, rhs1::AbstractVector)
    copyto!(sol1, rhs1)
    solve_system(system_solver.fact_cache, sol1)
    # TODO recover if fails - check issuccess
    return sol1
end
