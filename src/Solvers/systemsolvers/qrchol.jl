#=
Copyright 2018, Chris Coey and contributors

QR+Cholesky linear system solver
requires precomputed QR factorization of A'
solves linear system in naive.jl by first eliminating s, kap, and tau via the method in the symindef solver, then reducing the 3x3 symmetric indefinite system to a series of low-dimensional operations via a procedure similar to that described by S10.3 of
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf (the dominating subroutine is a positive definite linear solve with RHS of dimension n-p x 3)
=#

abstract type QRCholSystemSolver{T <: Real} <: SystemSolver{T} end

function solve_system(system_solver::QRCholSystemSolver{T}, solver::Solver{T}, sol::Matrix{T}, rhs::Matrix{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = n + p + q + 1

    # TODO in-place
    x = zeros(T, n, 3)
    @. @views x[:, 1:2] = rhs[1:n, :]
    @. @views x[:, 3] = -model.c
    x_sub1 = @view x[1:p, :]
    x_sub2 = @view x[(p + 1):end, :]

    y = zeros(T, p, 3)
    @. @views y[:, 1:2] = -rhs[n .+ (1:p), :]
    @. @views y[:, 3] = model.b

    z = zeros(T, q, 3)

    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        z_rows_k = (n + p) .+ idxs_k
        s_rows_k = tau_row .+ idxs_k
        zk12 = @view rhs[z_rows_k, :]
        sk12 = @view rhs[s_rows_k, :]
        hk = @view model.h[idxs_k]
        zk12_new = @view z[idxs_k, 1:2]
        zk3_new = @view z[idxs_k, 3]

        if Cones.use_dual(cone_k)
            zk12_temp = -zk12 - sk12 # TODO in place
            Cones.inv_hess_prod!(zk12_new, zk12_temp, cone_k)
            @. zk12_new /= solver.mu
            Cones.inv_hess_prod!(zk3_new, hk, cone_k)
            @. zk3_new /= solver.mu
        else
            Cones.hess_prod!(zk12_new, zk12, cone_k)
            @. zk12_new *= solver.mu
            @. zk12_new *= -1
            @. zk12_new -= sk12
            Cones.hess_prod!(zk3_new, hk, cone_k)
            @. zk3_new *= solver.mu
        end
    end

    ldiv!(solver.Ap_R', y)

    copyto!(system_solver.QpbxGHbz, x) # TODO can be avoided
    mul!(system_solver.QpbxGHbz, model.G', z, true, true)
    lmul!(solver.Ap_Q', system_solver.QpbxGHbz)

    copyto!(x_sub1, y)

    if !isempty(system_solver.Q2div)
        mul!(system_solver.GQ1x, system_solver.GQ1, y)
        @timeit solver.timer "block_hess_prod" block_hessian_product.(model.cones, system_solver.HGQ1x_k, system_solver.GQ1x_k, solver.mu)
        mul!(system_solver.Q2div, system_solver.GQ2', system_solver.HGQ1x, -1, true)

        @timeit solver.timer "solve_subsystem" solve_subsystem(system_solver, x_sub2, system_solver.Q2div)
    end

    lmul!(solver.Ap_Q, x)

    mul!(system_solver.Gx, model.G, x)
    @timeit solver.timer "block_hess_prod" block_hessian_product.(model.cones, system_solver.HGx_k, system_solver.Gx_k, solver.mu)

    @. z = system_solver.HGx - z

    if !isempty(y)
        copyto!(y, system_solver.Q1pbxGHbz)
        mul!(y, system_solver.GQ1', system_solver.HGx, -1, true)
        ldiv!(solver.Ap_R, y)
    end

    x3 = @view x[:, 3]
    y3 = @view y[:, 3]
    z3 = @view z[:, 3]
    x12 = @view x[:, 1:2]
    y12 = @view y[:, 1:2]
    z12 = @view z[:, 1:2]

    # lift to get tau
    # TODO maybe use higher precision here
    tau_denom = solver.mu / solver.tau / solver.tau - dot(model.c, x3) - dot(model.b, y3) - dot(model.h, z3)
    tau = @view sol[tau_row:tau_row, :]
    @. @views tau = rhs[tau_row:tau_row, :] + rhs[end:end, :]
    tau .+= model.c' * x12 + model.b' * y12 + model.h' * z12 # TODO in place
    @. tau /= tau_denom

    @. x12 += tau * x3
    @. y12 += tau * y3
    @. z12 += tau * z3

    @views sol[1:n, :] = x12
    @views sol[n .+ (1:p), :] = y12
    @views sol[(n + p) .+ (1:q), :] = z12

    # lift to get s and kap
    # TODO refactor below for use with symindef and qrchol methods
    # s = -G*x + h*tau - zrhs
    s = @view sol[(tau_row + 1):(end - 1), :]
    mul!(s, model.h, tau)
    mul!(s, model.G, sol[1:n, :], -one(T), true)
    @. @views s -= rhs[(n + p) .+ (1:q), :]

    # kap = -mu/(taubar^2)*tau + kaprhs
    @. @views sol[end:end, :] = -solver.mu / solver.tau * tau / solver.tau + rhs[end:end, :]

    return sol
end

function block_hessian_product(cone_k::Cones.Cone{T}, prod_k::AbstractMatrix{T}, arr_k::AbstractMatrix{T}, mu::T) where {T <: Real}
    if Cones.use_dual(cone_k)
        Cones.inv_hess_prod!(prod_k, arr_k, cone_k)
        @. prod_k /= mu
    else
        Cones.hess_prod!(prod_k, arr_k, cone_k)
        @. prod_k *= mu
    end
    return
end

#=
direct dense
=#

mutable struct QRCholDenseSystemSolver{T <: Real} <: QRCholSystemSolver{T}
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
        fact_cache::Union{DensePosDefCache{T}, DenseSymCache{T}} = DensePosDefCache{T}(),
        ) where {T <: Real}
        system_solver = new{T}()
        system_solver.fact_cache = fact_cache # TODO start with cholesky and then switch to BK if numerical issues
        return system_solver
    end
end

function load(system_solver::QRCholDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cone_idxs = model.cone_idxs

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
    system_solver.QpbxGHbz = Matrix{T}(undef, n, 3)
    system_solver.Q1pbxGHbz = view(system_solver.QpbxGHbz, 1:p, :)
    system_solver.Q2div = view(system_solver.QpbxGHbz, (p + 1):n, :)
    system_solver.GQ1x = Matrix{T}(undef, q, 3)
    system_solver.HGQ1x = similar(system_solver.GQ1x)
    system_solver.Gx = similar(system_solver.GQ1x)
    system_solver.HGx = similar(system_solver.Gx)
    system_solver.HGQ1x_k = [view(system_solver.HGQ1x, idxs, :) for idxs in cone_idxs]
    system_solver.GQ1x_k = [view(system_solver.GQ1x, idxs, :) for idxs in cone_idxs]
    system_solver.HGQ2_k = [view(system_solver.HGQ2, idxs, :) for idxs in cone_idxs]
    system_solver.GQ2_k = [view(system_solver.GQ2, idxs, :) for idxs in cone_idxs]
    system_solver.HGx_k = [view(system_solver.HGx, idxs, :) for idxs in cone_idxs]
    system_solver.Gx_k = [view(system_solver.Gx, idxs, :) for idxs in cone_idxs]

    load_matrix(system_solver.fact_cache, system_solver.lhs1) # overwrite lhs1 with new factorization each time update_fact is called

    return system_solver
end

# TODO move to dense.jl?
outer_prod(UGQ2::AbstractMatrix{T}, lhs1::AbstractMatrix{T}) where {T <: LinearAlgebra.BlasReal} = BLAS.syrk!('U', 'T', true, UGQ2, true, lhs1)
outer_prod(UGQ2::AbstractMatrix{T}, lhs1::AbstractMatrix{T}) where {T <: Real} = mul!(lhs1, UGQ2', UGQ2, true, true)

function update_fact(system_solver::QRCholDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    isempty(system_solver.Q2div) && return system_solver
    model = solver.model

    # TODO use dispatch
    # TODO faster if only do one syrk from the first block of indices and one mul from the second block
    system_solver.lhs1.data .= 0
    sqrtmu = sqrt(solver.mu)
    for (cone_k, prod_k, arr_k) in zip(model.cones, system_solver.HGQ2_k, system_solver.GQ2_k)
        if hasfield(typeof(cone_k), :hess_fact_cache) && cone_k.hess_fact_cache isa DenseSymCache{T}
            block_hessian_product(cone_k, prod_k, arr_k, solver.mu)
            mul!(system_solver.lhs1.data, arr_k', prod_k, true, true)
        else
            if Cones.use_dual(cone_k)
                Cones.inv_hess_sqrt_prod!(prod_k, arr_k, cone_k)
                prod_k ./= sqrtmu
            else
                Cones.hess_sqrt_prod!(prod_k, arr_k, cone_k)
                prod_k .*= sqrtmu
            end
            outer_prod(prod_k, system_solver.lhs1.data)
        end
    end

    if !update_fact(system_solver.fact_cache, system_solver.lhs1)
        if system_solver.fact_cache isa DensePosDefCache{T}
            @warn("Switching QRChol solver from Cholesky to Bunch Kaufman")
            system_solver.fact_cache = DenseSymCache{T}()
            load_matrix(system_solver.fact_cache, system_solver.lhs1)
        else
            system_solver.lhs1 += sqrt(eps(T)) * I # attempt recovery # TODO make more efficient
        end
        if !update_fact(system_solver.fact_cache, system_solver.lhs1)
            @warn("QRChol Bunch Kaufman factorization failed")
        end
    end

    return system_solver
end

function solve_subsystem(system_solver::QRCholDenseSystemSolver, sol1::AbstractMatrix, rhs1::AbstractMatrix)
    copyto!(sol1, rhs1)
    inv_prod(system_solver.fact_cache, sol1)
    return sol1
end
