#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

naive+elimination linear system solver
solves linear system in naive.jl via the following procedure

eliminate s
-G*x + h*tau - s = zrhs
so if using primal barrier
z_k + mu*H_k*s_k = srhs_k --> s_k = (mu*H_k)\(srhs_k - z_k)
-->
-G_k*x + (mu*H_k)\z_k + h_k*tau = zrhs_k + (mu*H_k)\srhs_k
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



# TODO delete - no longer needed

# helpers for sparse linear system solvers
function add_I_J_V(
    offset::Int,
    Is::Vector{<:Integer},
    Js::Vector{<:Integer},
    Vs::Vector{Float64},
    start_row::Int,
    start_col::Int,
    vec::Vector{Float64},
    trans::Bool,
    )
    n = length(vec)
    idxs = offset:(offset + n - 1)
    if !isempty(vec)
        if trans
            @. Is[idxs] = start_row + 1
            @. Js[idxs] = (start_col + 1):(start_col + n)
        else
            @. Is[idxs] = (start_row + 1):(start_row + n)
            @. Js[idxs] = start_col + 1
        end
        @. Vs[idxs] = vec
    end
    return offset + n
end

function add_I_J_V(
    offset::Int,
    Is::Vector{<:Integer},
    Js::Vector{<:Integer},
    Vs::Vector{Float64},
    start_row::Int,
    start_col::Int,
    mat::SparseMatrixCSC,
    trans::Bool,
    )
    for j in 1:size(mat, 2)
        col_idxs = mat.colptr[j]:(mat.colptr[j + 1] - 1)
        rows = view(mat.rowval, col_idxs)
        m = length(rows)
        idxs = offset:(offset + m - 1)
        if trans
            @. Is[idxs] = start_row + j
            @. Js[idxs] = start_col + rows
        else
            @. Is[idxs] = start_row + rows
            @. Js[idxs] = start_col + j
        end
        Vs[idxs] .= view(mat.nzval, col_idxs)
        offset += m
    end
    return offset
end

function add_I_J_V(
    offset::Int,
    Is::Vector{<:Integer},
    Js::Vector{<:Integer},
    Vs::Vector{Float64},
    start_row::Int,
    start_col::Int,
    cone::Cones.Cone,
    use_inv::Bool,
    lower_only::Bool = false,
    )
    for j in 1:Cones.dimension(cone)
        nz_rows = (use_inv ? Cones.inv_hess_nz_idxs_j(cone, j, lower_only) : Cones.hess_nz_idxs_j(cone, j, lower_only))
        n = length(nz_rows)
        idxs = offset:(offset + n - 1)
        @. Is[idxs] = start_row + nz_rows
        @. Js[idxs] = j + start_col
        @. Vs[idxs] = 1
        offset += n
    end
    return offset
end



mutable struct NaiveElimSparseSystemSolver{T <: Real} <: NaiveElimSystemSolver{T}
    use_inv_hess::Bool
    tau_row::Int
    lhs4::SparseMatrixCSC # TODO CSC type will depend on factor cache Int type
    rhs4::Matrix{T}
    sol4::Matrix{T}
    fact_cache::SparseNonSymCache{T}
    hess_idxs

    function NaiveElimSparseSystemSolver{T}(;
        use_inv_hess::Bool = true,
        fact_cache::SparseNonSymCache{Float64} = SparseNonSymCache(),
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
    system_solver.sol4 = zeros(system_solver.tau_row, 2)
    system_solver.rhs4 = similar(system_solver.sol4)

    # TODO remove
    model.A = sparse(model.A)
    model.G = sparse(model.G)
    dropzeros!(model.A)
    dropzeros!(model.G)

    # count the number of nonzeros we will have in the lhs
    hess_nnzs = sum(Cones.hess_nnzs(cone_k, false) for cone_k in model.cones)
    nnzs = 2 * (nnz(model.A) + nnz(model.G) + n + p + q) + hess_nnzs + 1

    # NOTE if more caches are added, Int32 will need to be updated so that it is not hard coded
    Is = Vector{Int32}(undef, nnzs)
    Js = Vector{Int32}(undef, nnzs)
    Vs = Vector{Float64}(undef, nnzs)

    # compute the starting rows/columns for each known block in the lhs
    rc1 = 0
    rc2 = n
    rc3 = n + p
    rc4 = n + p + q

    # count of nonzeros added so far
    offset = 1
    # update I, J, V while adding A and G blocks to the lhs
    offset = add_I_J_V(offset, Is, Js, Vs, rc1, rc2, model.A, true)
    offset = add_I_J_V(offset, Is, Js, Vs, rc1, rc3, model.G, true)
    offset = add_I_J_V(offset, Is, Js, Vs, rc1, rc4, model.c, false)
    offset = add_I_J_V(offset, Is, Js, Vs, rc2, rc1, -model.A, false)
    offset = add_I_J_V(offset, Is, Js, Vs, rc2, rc4, model.b, false)
    offset = add_I_J_V(offset, Is, Js, Vs, rc3, rc1, -model.G, false)
    offset = add_I_J_V(offset, Is, Js, Vs, rc3, rc4, model.h, false)
    offset = add_I_J_V(offset, Is, Js, Vs, rc4, rc1, -model.c, true)
    offset = add_I_J_V(offset, Is, Js, Vs, rc4, rc2, -model.b, true)
    offset = add_I_J_V(offset, Is, Js, Vs, rc4, rc3, -model.h, true)
    offset = add_I_J_V(offset, Is, Js, Vs, rc4, rc4, [1.0], false) # mu / tau / tau placeholder

    nz_rows_added = 0
    @timeit solver.timer "setup_hess" for (k, cone_k) in enumerate(model.cones)
        cone_dim = Cones.dimension(cone_k)
        rows = n + p + nz_rows_added
        offset = add_I_J_V(offset, Is, Js, Vs, rows, rows, cone_k, !Cones.use_dual(cone_k), false)
        nz_rows_added += cone_dim
    end

    @assert offset == nnzs + 1
    dim = n + p + q + 1
    @timeit solver.timer "build_sparse" system_solver.lhs4 = sparse(Is, Js, Vs, Int32(dim), Int32(dim))
    lhs4 = system_solver.lhs4

    # cache indices of placeholders of Hessians
    system_solver.hess_idxs = [Vector{UnitRange}(undef, Cones.dimension(cone_k)) for cone_k in model.cones]
    row = col = n + p + 1
    for (k, cone_k) in enumerate(model.cones)
        cone_dim = Cones.dimension(cone_k)
        for j in 1:cone_dim
            # get list of nonzero rows in the current column of the LHS
            col_idx_start = lhs4.colptr[col]
            col_idx_end = lhs4.colptr[col + 1] - 1
            nz_rows = lhs4.rowval[col_idx_start:col_idx_end]
            # nonzero rows in column j of the hessian
            nz_hess_indices = Cones.hess_nz_idxs_j(cone_k, j, false)
            # index corresponding to first nonzero Hessian element of the current column of the LHS
            offset_in_row = findfirst(x -> x == row + nz_hess_indices[1] - 1, nz_rows)
            # indices of nonzero values for cone k column j
            system_solver.hess_idxs[k][j] = col_idx_start + offset_in_row - nz_hess_indices[1] .+ nz_hess_indices .- 1
            # move to the next column
            col += 1
        end
        row += cone_dim
    end

    return system_solver
end

function update_fact(system_solver::NaiveElimSparseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    for (k, cone_k) in enumerate(solver.model.cones)
        @timeit solver.timer "update_hess" H = (Cones.use_dual(cone_k) ? Cones.hess(cone_k) : Cones.inv_hess(cone_k))
        for j in 1:Cones.dimension(cone_k)
            nz_rows = Cones.hess_nz_idxs_j(cone_k, j, false)
            @views copyto!(system_solver.lhs4.nzval[system_solver.hess_idxs[k][j]], H[nz_rows, j])
        end
    end
    # update mu / tau / tau placeholder, which is the last nonzero added
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
        fact_cache = nothing,
        ) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_inv_hess = use_inv_hess
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
        zeros(T,n,n)  model.A'      model.G'              model.c;
        -model.A      zeros(T,p,p)  zeros(T,p,q)          model.b;
        -model.G      zeros(T,q,p)  Matrix(one(T)*I,q,q)  model.h;
        -model.c'     -model.b'     -model.h'             one(T);
        ]
    system_solver.lhs4 = similar(system_solver.lhs4_copy)
    # system_solver.fact_cache = HypLUSolveCache(system_solver.sol, system_solver.lhs4, rhs4)

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
            lhs4[z_rows_k, z_rows_k] .= Cones.inv_hess(cone_k)
        else
            # -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
            @views Cones.hess_prod!(lhs4[z_rows_k, 1:n], model.G[idxs_k, :], cone_k)
            @. lhs4[z_rows_k, 1:n] *= -1
            @views Cones.hess_prod!(lhs4[z_rows_k, end], model.h[idxs_k], cone_k)
        end
    end

    system_solver.fact_cache = lu!(system_solver.lhs4) # TODO use wrapped lapack function

    return system_solver
end

function solve_subsystem(system_solver::NaiveElimDenseSystemSolver{T}, solver::Solver{T}, sol4::Matrix{T}, rhs4::Matrix{T}) where {T <: Real}
    ldiv!(sol4, system_solver.fact_cache, rhs4)
    return sol4
end
