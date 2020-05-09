#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

naive linear system solver

6x6 nonsymmetric system in (x, y, z, tau, s, kap):
A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
-G*x + h*tau - s = zrhs
-c'*x - b'*y - h'*z - kap = taurhs
(pr bar) z_k + mu*H_k*s_k = srhs_k
(du bar) mu*H_k*z_k + s_k = srhs_k
mu/(taubar^2)*tau + kap = kaprhs
=#

abstract type NaiveSystemSolver{T <: Real} <: SystemSolver{T} end

#=
indirect
TODO
- precondition
- optimize operations
- tune number of restarts and tolerances etc, ensure initial point in sol helps
- fix IterativeSolvers so that methods can take matrix RHS
=#

# mutable struct NaiveIndirectSystemSolver{T <: Real} <: NaiveSystemSolver{T}
#     lhs6::BlockMatrix{T}
#     NaiveIndirectSystemSolver{T}() where {T <: Real} = new{T}()
# end
#
# function load(system_solver::NaiveIndirectSystemSolver{T}, solver::Solver{T}) where {T <: Real}
#     model = solver.model
#     (n, p, q) = (model.n, model.p, model.q)
#     cones = model.cones
#     cone_idxs = model.cone_idxs
#     tau_row = n + p + q + 1
#     dim = tau_row + q + 1
#
#     # setup block LHS
#     x_idxs = 1:n
#     y_idxs = n .+ (1:p)
#     z_idxs = (n + p) .+ (1:q)
#     tau_idxs = tau_row:tau_row
#     s_idxs = tau_row .+ (1:q)
#     kap_idxs = dim:dim
#
#     k_len = 2 * length(cones)
#     cone_rows = Vector{UnitRange{Int}}(undef, k_len)
#     cone_cols = Vector{UnitRange{Int}}(undef, k_len)
#     cone_blocks = Vector{Any}(undef, k_len)
#     for (k, cone_k) in enumerate(cones)
#         idxs_k = model.cone_idxs[k]
#         rows = tau_row .+ idxs_k
#         k1 = 2k - 1
#         k2 = 2k
#         cone_rows[k1] = cone_rows[k2] = rows
#         cone_cols[k1] = (n + p) .+ idxs_k
#         cone_cols[k2] = rows
#         if Cones.use_dual_barrier(cone_k)
#             cone_blocks[k1] = cone_k
#             cone_blocks[k2] = I
#         else
#             cone_blocks[k1] = I
#             cone_blocks[k2] = cone_k
#         end
#     end
#
#     system_solver.lhs6 = BlockMatrix{T}(dim, dim,
#         [cone_blocks...,
#             model.A', model.G', reshape(model.c, :, 1),
#             -model.A, reshape(model.b, :, 1),
#             -model.G, reshape(model.h, :, 1), -I,
#             -model.c', -model.b', -model.h', -ones(T, 1, 1),
#             solver, ones(T, 1, 1)],
#         [cone_rows...,
#             x_idxs, x_idxs, x_idxs,
#             y_idxs, y_idxs,
#             z_idxs, z_idxs, z_idxs,
#             tau_idxs, tau_idxs, tau_idxs, tau_idxs,
#             kap_idxs, kap_idxs],
#         [cone_cols...,
#             y_idxs, z_idxs, tau_idxs,
#             x_idxs, tau_idxs,
#             x_idxs, tau_idxs, s_idxs,
#             x_idxs, y_idxs, z_idxs, kap_idxs,
#             tau_idxs, kap_idxs],
#         )
#
#     return system_solver
# end
#
# update_lhs(system_solver::NaiveIndirectSystemSolver, solver::Solver) = system_solver
#
# function solve_system(system_solver::NaiveIndirectSystemSolver, solver::Solver, sol6::Matrix, rhs6::Matrix)
#     for j in 1:size(rhs6, 2)
#         rhs_j = view(rhs6, :, j)
#         sol_j = view(sol6, :, j)
#         IterativeSolvers.gmres!(sol_j, system_solver.lhs6, rhs_j, restart = size(rhs6, 1))
#     end
#     return sol6
# end

#=
direct sparse
=#

mutable struct NaiveSparseSystemSolver{T <: Real} <: NaiveSystemSolver{T}
    lhs6::SparseMatrixCSC
    hess_idxs::Vector
    mtt_idx::Int
    fact_cache::SparseNonSymCache{T}
    function NaiveSparseSystemSolver{Float64}(; fact_cache::SparseNonSymCache{Float64} = SparseNonSymCache{Float64}())
        s = new{Float64}()
        s.fact_cache = fact_cache
        return s
    end
end

function load(system_solver::NaiveSparseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.fact_cache.analyzed = false
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = n + p + q + 1
    cones = model.cones
    cone_idxs = model.cone_idxs

    # form sparse LHS without Hessians in s row
    lhs6 = T[
        spzeros(T, n, n)  model.A'          model.G'                  model.c        spzeros(T, n, q)           spzeros(T, n);
        -model.A          spzeros(T, p, p)  spzeros(T, p, q)          model.b        spzeros(T, p, q)           spzeros(T, p);
        -model.G          spzeros(T, q, p)  spzeros(T, q, q)          model.h        sparse(-one(T) * I, q, q)  spzeros(T, q);
        -model.c'         -model.b'         -model.h'                 zero(T)        spzeros(T, 1, q)           -one(T);
        spzeros(T, q, n)  spzeros(T, q, p)  sparse(one(T) * I, q, q)  spzeros(T, q)  sparse(one(T) * I, q, q)   spzeros(T, q);
        spzeros(T, 1, n)  spzeros(T, 1, p)  spzeros(T, 1, q)          one(T)         spzeros(T, 1, q)           one(T);
        ]
    @assert issparse(lhs6)
    dropzeros!(lhs6)
    (Is, Js, Vs) = findnz(lhs6)

    # add I, J, V for Hessians
    if isempty(cones)
        hess_nz_total = 0
    else
        hess_nz_total = sum(Cones.hess_nz_count(cone_k) for cone_k in cones)
    end
    H_Is = Vector{Int}(undef, hess_nz_total)
    H_Js = Vector{Int}(undef, hess_nz_total)
    offset = 1
    for (cone_k, idxs_k) in zip(cones, cone_idxs)
        z_start_k = n + p + first(idxs_k) - 1
        s_start_k = tau_row + first(idxs_k) - 1
        H_start_k = Cones.use_dual_barrier(cone_k) ? z_start_k : s_start_k
        for j in 1:Cones.dimension(cone_k)
            nz_rows_kj = s_start_k .+ Cones.hess_nz_idxs_col(cone_k, j)
            len_kj = length(nz_rows_kj)
            IJV_idxs = offset:(offset + len_kj - 1)
            offset += len_kj
            @. H_Is[IJV_idxs] = nz_rows_kj
            @. H_Js[IJV_idxs] = H_start_k + j
        end
    end
    @assert offset == hess_nz_total + 1
    append!(Is, H_Is)
    append!(Js, H_Js)
    append!(Vs, ones(T, hess_nz_total))

    # prefer conversions of integer types to happen here than inside external wrappers
    dim = size(lhs6, 1)
    # integer type supported by the sparse system solver library to be used
    Ti = int_type(system_solver.fact_cache)
    Is = convert(Vector{Ti}, Is)
    Js = convert(Vector{Ti}, Js)
    lhs6 = system_solver.lhs6 = sparse(Is, Js, Vs, dim, dim)

    # cache indices of nonzeros of Hessians in sparse LHS nonzeros vector
    system_solver.hess_idxs = [Vector{Union{UnitRange, Vector{Int}}}(undef, Cones.dimension(cone_k)) for cone_k in cones]
    for (k, cone_k) in enumerate(cones)
        idxs_k = cone_idxs[k]
        z_start_k = n + p + first(idxs_k) - 1
        s_start_k = tau_row + first(idxs_k) - 1
        H_start_k = Cones.use_dual_barrier(cone_k) ? z_start_k : s_start_k
        for j in 1:Cones.dimension(cone_k)
            col = H_start_k + j
            # get nonzero rows in the current column of the LHS
            col_idx_start = lhs6.colptr[col]
            nz_rows = lhs6.rowval[col_idx_start:(lhs6.colptr[col + 1] - 1)]
            # get nonzero rows in column j of the Hessian
            nz_hess_indices = Cones.hess_nz_idxs_col(cone_k, j)
            # get index corresponding to first nonzero Hessian element of the current column of the LHS
            first_H = findfirst(isequal(s_start_k + first(nz_hess_indices)), nz_rows)
            # indices of nonzero values for cone k column j
            system_solver.hess_idxs[k][j] = (col_idx_start + first_H - 2) .+ (1:length(nz_hess_indices))
        end
    end

    # get mu/tau/tau index
    system_solver.mtt_idx = lhs6.colptr[tau_row + 1] - 1

    return system_solver
end

function update_lhs(system_solver::NaiveSparseSystemSolver, solver::Solver)
    for (k, cone_k) in enumerate(solver.model.cones)
        H_k = Cones.hess(cone_k)
        for j in 1:Cones.dimension(cone_k)
            nz_rows = Cones.hess_nz_idxs_col(cone_k, j)
            @. @views system_solver.lhs6.nzval[system_solver.hess_idxs[k][j]] = solver.mu * H_k[nz_rows, j]
        end
    end
    system_solver.lhs6.nzval[system_solver.mtt_idx] = solver.mu / solver.tau / solver.tau

    @timeit solver.timer "update_fact" update_fact(system_solver.fact_cache, system_solver.lhs6)

    return system_solver
end

function solve_system(system_solver::NaiveSparseSystemSolver, solver::Solver, sol6::Vector, rhs6::Vector)
    inv_prod(system_solver.fact_cache, sol6, system_solver.lhs6, rhs6)
    return sol6
end

#=
direct dense
=#

mutable struct NaiveDenseSystemSolver{T <: Real} <: NaiveSystemSolver{T}
    tau_row::Int
    lhs6::Matrix{T}
    lhs6_H_k::Vector
    fact_cache::DenseNonSymCache{T}
    function NaiveDenseSystemSolver{T}(; fact_cache::DenseNonSymCache{T} = DenseNonSymCache{T}()) where {T <: Real}
        system_solver = new{T}()
        system_solver.fact_cache = fact_cache
        return system_solver
    end
end

function load(system_solver::NaiveDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs
    system_solver.tau_row = n + p + q + 1

    system_solver.lhs6 = T[
        zeros(T, n, n)  model.A'        model.G'                  model.c      zeros(T, n, q)             zeros(T, n);
        -model.A        zeros(T, p, p)  zeros(T, p, q)            model.b      zeros(T, p, q)             zeros(T, p);
        -model.G        zeros(T, q, p)  zeros(T, q, q)            model.h      Matrix(-one(T) * I, q, q)  zeros(T, q);
        -model.c'       -model.b'       -model.h'                 zero(T)      zeros(T, 1, q)             -one(T);
        zeros(T, q, n)  zeros(T, q, p)  Matrix(one(T) * I, q, q)  zeros(T, q)  Matrix(one(T) * I, q, q)   zeros(T, q);
        zeros(T, 1, n)  zeros(T, 1, p)  zeros(T, 1, q)            one(T)       zeros(T, 1, q)             one(T);
        ]

    function view_H_k(cone_k, idxs_k)
        rows = system_solver.tau_row .+ idxs_k
        cols = Cones.use_dual_barrier(cone_k) ? (n + p) .+ idxs_k : rows
        return view(system_solver.lhs6, rows, cols)
    end
    system_solver.lhs6_H_k = [view_H_k(cone_k, idxs_k) for (cone_k, idxs_k) in zip(cones, cone_idxs)]

    load_matrix(system_solver.fact_cache, system_solver.lhs6)

    return system_solver
end

function update_lhs(system_solver::NaiveDenseSystemSolver, solver::Solver)
    # for (cone_k, lhs6_H_k) in zip(solver.model.cones, system_solver.lhs6_H_k)
    for k in eachindex(system_solver.lhs6_H_k)
        cone_k = solver.model.cones[k]
        lhs6_H_k = system_solver.lhs6_H_k[k]
        if Cones.use_scaling(cone_k)
            scal_hess = Cones.scal_hess(cone_k, solver.mu)
            @. lhs6_H_k = scal_hess
        else
            H_k = Cones.hess(cone_k)
            @. lhs6_H_k = solver.mu * H_k
        end
    end
    system_solver.lhs6[end, system_solver.tau_row] = solver.kap / solver.tau

    @timeit solver.timer "update_fact" update_fact(system_solver.fact_cache, system_solver.lhs6)

    return system_solver
end

function solve_system(system_solver::NaiveDenseSystemSolver, solver::Solver, sol6::Vector, rhs6::Vector)
    copyto!(sol6, rhs6)
    inv_prod(system_solver.fact_cache, sol6)
    return sol6
end
