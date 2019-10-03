#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

naive+elimination linear system solver
solves linear system in naive.jl via the following procedure

eliminate s
-G*x + h*tau - s = zrhs
so if using primal barrier
z_k + mu*H_k*s_k = srhs_k --> s_k = (mu*H_k)\(srhs_k - z_k)
-->
-G_k*x + (mu*H_k)\z_k + h_k*tau = zrhs_k + (mu*H_k)\srhs_k (if use_inv_hess = true)
-->
-mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k (if use_inv_hess = false)
or if using dual barrier
mu*H_k*z_k + s_k = srhs_k --> s_k = srhs_k - mu*H_k*z_k
-->
-G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k

eliminate kap
-c'x - b'y - h'z - kap = taurhs
so
mu/(taubar^2)*tau + kap = kaprhs --> kap = kaprhs - mu/(taubar^2)*tau
-->
-c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs

4x4 nonsymmetric system in (x, y, z, tau):
A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
(pr bar) -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
(du bar) -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
-c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs
=#

abstract type NaiveElimSystemSolver{T <: Real} <: SystemSolver{T} end

function solve_system(system_solver::NaiveElimSystemSolver{T}, solver::Solver{T}, sol::Matrix{T}, rhs::Matrix{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = system_solver.tau_row

    sol4 = system_solver.sol4
    rhs4 = system_solver.rhs4
    @views copyto!(rhs4, rhs[1:tau_row, :])

    for (k, cone_k) in enumerate(model.cones)
        z_rows_k = (n + p) .+ model.cone_idxs[k]
        s_rows_k = (q + 1) .+ z_rows_k
        if Cones.use_dual(cone_k)
            # -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
            @. @views rhs4[z_rows_k, :] += rhs[s_rows_k, :]
        elseif system_solver.use_inv_hess
            # -G_k*x + (mu*H_k)\z_k + h_k*tau = zrhs_k + (mu*H_k)\srhs_k
            @views Cones.inv_hess_prod!(rhs4[z_rows_k, :], rhs[s_rows_k, :], cone_k)
            @. @views rhs4[z_rows_k, :] += rhs[z_rows_k, :]
        else
            # -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
            @views Cones.hess_prod!(rhs4[z_rows_k, :], rhs[z_rows_k, :], cone_k)
            @. @views rhs4[z_rows_k, :] += rhs[s_rows_k, :]
        end
    end
    # -c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs
    @. @views rhs4[end, :] += rhs[end, :]

    solve_subsystem(system_solver, solver, sol4, rhs4)
    @views copyto!(sol[1:tau_row, :], sol4)

    # lift to get s and kap
    tau = sol4[end:end, :]

    # TODO refactor below for use with symindef and qrchol methods
    # s = -G*x + h*tau - zrhs
    s = @view sol[(tau_row + 1):(end - 1), :]
    mul!(s, model.h, tau)
    x = @view sol[1:n, :]
    mul!(s, model.G, x, -one(T), true)
    @. @views s -= rhs[(n + p) .+ (1:q), :]

    # kap = -mu/(taubar^2)*tau + kaprhs
    @. @views sol[end:end, :] = -solver.mu / solver.tau * tau / solver.tau + rhs[end:end, :]

    return sol
end

#=
direct sparse
=#

mutable struct NaiveElimSparseSystemSolver{T <: Real} <: NaiveElimSystemSolver{T}
    use_inv_hess::Bool
    tau_row::Int
    lhs4::SparseMatrixCSC # TODO CSC type will depend on factor cache Int type
    rhs4::Matrix{T}
    sol4::Matrix{T}
    fact_cache::SparseNonSymCache{T}
    hess_idxs::Vector{Vector{Union{UnitRange, Vector{Int}}}}

    function NaiveElimSparseSystemSolver{T}(;
        use_inv_hess::Bool = true,
        fact_cache::SparseNonSymCache{Float64} = SparseNonSymCache{Float64}(),
        ) where {T <: Real}
        system_solver = new{T}()
        if !use_inv_hess
            @warn("SymIndefSparseSystemSolver is not implemented with `use_inv_hess` set to `false`, using `true` instead.")
        end
        system_solver.use_inv_hess = use_inv_hess
        system_solver.fact_cache = fact_cache
        return system_solver
    end
end

function load(system_solver::NaiveElimSparseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.fact_cache.analyzed = false
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    system_solver.tau_row = n + p + q + 1
    cones = model.cones
    cone_idxs = model.cone_idxs

    system_solver.sol4 = zeros(system_solver.tau_row, 2)
    system_solver.rhs4 = similar(system_solver.sol4)

    # form sparse LHS without Hessians and inverse Hessians in z/z block
    lhs4 = T[
        spzeros(T, n, n)  model.A'          model.G'                  model.c;
        -model.A          spzeros(T, p, p)  spzeros(T, p, q)          model.b;
        -model.G          spzeros(T, q, p)  sparse(one(T) * I, q, q)  model.h;
        -model.c'         -model.b'         -model.h'                 one(T);
        ]
    @assert issparse(lhs4)
    dropzeros!(lhs4)
    (Is, Js, Vs) = findnz(lhs4)

    # add I, J, V for Hessians and inverse Hessians
    hess_nz_total = sum(Cones.use_dual(cone_k) ? Cones.hess_nz_count(cone_k, false) : Cones.inv_hess_nz_count(cone_k, false) for cone_k in cones)
    H_Is = Vector{Int}(undef, hess_nz_total)
    H_Js = Vector{Int}(undef, hess_nz_total)
    offset = 1
    for (k, cone_k) in enumerate(cones)
        cone_idxs_k = cone_idxs[k]
        z_start_k = n + p + first(cone_idxs_k) - 1
        for j in 1:Cones.dimension(cone_k)
            nz_rows_kj = z_start_k .+ (Cones.use_dual(cone_k) ? Cones.hess_nz_idxs_col(cone_k, j, false) : Cones.inv_hess_nz_idxs_col(cone_k, j, false))
            len_kj = length(nz_rows_kj)
            IJV_idxs = offset:(offset + len_kj - 1)
            offset += len_kj
            @. H_Is[IJV_idxs] = nz_rows_kj
            @. H_Js[IJV_idxs] = z_start_k + j
        end
    end
    @assert offset == hess_nz_total + 1
    append!(Is, H_Is)
    append!(Js, H_Js)
    append!(Vs, ones(T, hess_nz_total))

    # prefer conversions of integer types to happen here than inside external wrappers
    dim = size(lhs4, 1)
    # integer type supported by the sparse system solver library to be used
    Ti = int_type(system_solver.fact_cache)
    Is = convert(Vector{Ti}, Is)
    Js = convert(Vector{Ti}, Js)
    lhs4 = system_solver.lhs4 = sparse(Is, Js, Vs, dim, dim)

    # cache indices of nonzeros of Hessians and inverse Hessians in sparse LHS nonzeros vector
    system_solver.hess_idxs = [Vector{Union{UnitRange, Vector{Int}}}(undef, Cones.dimension(cone_k)) for cone_k in cones]
    for (k, cone_k) in enumerate(cones)
        cone_idxs_k = cone_idxs[k]
        z_start_k = n + p + first(cone_idxs_k) - 1
        for j in 1:Cones.dimension(cone_k)
            col = z_start_k + j
            # get nonzero rows in the current column of the LHS
            col_idx_start = lhs4.colptr[col]
            nz_rows = lhs4.rowval[col_idx_start:(lhs4.colptr[col + 1] - 1)]
            # get nonzero rows in column j of the Hessian or inverse Hessian
            nz_hess_indices = (Cones.use_dual(cone_k) ? Cones.hess_nz_idxs_col(cone_k, j, false) : Cones.inv_hess_nz_idxs_col(cone_k, j, false))
            # get index corresponding to first nonzero Hessian element of the current column of the LHS
            first_H = findfirst(isequal(z_start_k + first(nz_hess_indices)), nz_rows)
            # indices of nonzero values for cone k column j
            system_solver.hess_idxs[k][j] = (col_idx_start + first_H - 2) .+ (1:length(nz_hess_indices))
        end
    end

    return system_solver
end

function update_fact(system_solver::NaiveElimSparseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    for (k, cone_k) in enumerate(solver.model.cones)
        H = (Cones.use_dual(cone_k) ? Cones.hess(cone_k) : Cones.inv_hess(cone_k))
        for j in 1:Cones.dimension(cone_k)
            nz_rows = (Cones.use_dual(cone_k) ? Cones.hess_nz_idxs_col(cone_k, j, false) : Cones.inv_hess_nz_idxs_col(cone_k, j, false))
            @views copyto!(system_solver.lhs4.nzval[system_solver.hess_idxs[k][j]], H[nz_rows, j])
        end
    end
    system_solver.lhs4.nzval[end] = solver.mu / solver.tau / solver.tau

    update_sparse_fact(system_solver.fact_cache, system_solver.lhs4)

    return system_solver
end

function solve_subsystem(system_solver::NaiveElimSparseSystemSolver{T}, solver::Solver{T}, sol4::Matrix{T}, rhs4::Matrix{T}) where {T <: Real}
    @timeit solver.timer "solve_sparse_system" solve_sparse_system(system_solver.fact_cache, sol4, system_solver.lhs4, rhs4)
    return sol4
end

#=
direct dense
=#

mutable struct NaiveElimDenseSystemSolver{T <: Real} <: NaiveElimSystemSolver{T}
    use_inv_hess::Bool
    tau_row::Int
    lhs4::Matrix{T}
    lhs4_copy::Matrix{T}
    rhs4::Matrix{T}
    sol4::Matrix{T}
    fact_cache
    function NaiveElimDenseSystemSolver{T}(;
        use_inv_hess::Bool = true,
        fact_cache::DenseNonSymCache{T} = DenseNonSymCache{T}(),
        ) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_inv_hess = use_inv_hess
        system_solver.fact_cache = fact_cache
        return system_solver
    end
end

function load(system_solver::NaiveElimDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    system_solver.tau_row = n + p + q + 1
    system_solver.sol4 = zeros(T, system_solver.tau_row, 2)
    system_solver.rhs4 = similar(system_solver.sol4)

    system_solver.lhs4_copy = T[
        zeros(T, n, n)  model.A'        model.G'                  model.c;
        -model.A        zeros(T, p, p)  zeros(T, p, q)            model.b;
        -model.G        zeros(T, q, p)  Matrix(one(T) * I, q, q)  model.h;
        -model.c'       -model.b'       -model.h'                 one(T);
        ]
    system_solver.lhs4 = similar(system_solver.lhs4_copy)

    load_dense_matrix(system_solver.fact_cache, system_solver.lhs4)

    return system_solver
end

function update_fact(system_solver::NaiveElimDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p) = (model.n, model.p)
    lhs4 = system_solver.lhs4

    copyto!(lhs4, system_solver.lhs4_copy)
    lhs4[end, end] = solver.mu / solver.tau / solver.tau

    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        z_rows_k = (n + p) .+ idxs_k
        if Cones.use_dual(cone_k)
            # -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
            lhs4[z_rows_k, z_rows_k] .= Cones.hess(cone_k)
        elseif system_solver.use_inv_hess
            # -G_k*x + (mu*H_k)\z_k + h_k*tau = zrhs_k + (mu*H_k)\srhs_k
            lhs4[z_rows_k, z_rows_k] .= Cones.inv_hess(cone_k)
        else
            # -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
            @views Cones.hess_prod!(lhs4[z_rows_k, 1:n], model.G[idxs_k, :], cone_k)
            @. lhs4[z_rows_k, 1:n] *= -1
            @views Cones.hess_prod!(lhs4[z_rows_k, end], model.h[idxs_k], cone_k)
        end
    end

    reset_fact(system_solver.fact_cache)

    return system_solver
end

function solve_subsystem(system_solver::NaiveElimDenseSystemSolver{T}, solver::Solver{T}, sol4::Matrix{T}, rhs4::Matrix{T}) where {T <: Real}
    @timeit solver.timer "solve_dense_system" if !solve_dense_system(system_solver.fact_cache, sol4, system_solver.lhs4, rhs4)
        # TODO recover somehow
        @warn("numerical failure: could not solve linear system")
    end
    return sol4
end
