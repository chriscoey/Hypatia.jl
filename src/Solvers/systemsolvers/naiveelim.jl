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
tau + taubar/kapbar*kap = kaprhs --> kap = kapbar/taubar*(kaprhs - tau) # Nesterov-Todd
-->
-c'x - b'y - h'z + kapbar/taubar*tau = taurhs + kapbar/taubar*kaprhs

4x4 nonsymmetric system in (x, y, z, tau):
A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
(pr bar) -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
(du bar) -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
-c'x - b'y - h'z + kapbar/taubar*tau = taurhs + kapbar/taubar*kaprhs
=#

abstract type NaiveElimSystemSolver{T <: Real} <: SystemSolver{T} end

function solve_system(system_solver::NaiveElimSystemSolver{T}, solver::Solver{T}, sol::Vector{T}, rhs::Vector{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)

    sol4 = system_solver.sol4
    rhs4 = system_solver.rhs4
    dim4 = size(sol4, 1)
    @views copyto!(rhs4, rhs[1:dim4])

    for (k, cone_k) in enumerate(model.cones)
        z_rows_k = (n + p) .+ model.cone_idxs[k]
        s_rows_k = (q + 1) .+ z_rows_k
        if Cones.use_dual(cone_k) # no scaling
            # -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
            @. @views rhs4[z_rows_k] += rhs[s_rows_k]
        elseif system_solver.use_inv_hess
            # -G_k*x + (mu*H_k)\z_k + h_k*tau = zrhs_k + (mu*H_k)\srhs_k
            @views Cones.inv_hess_prod!(rhs4[z_rows_k], rhs[s_rows_k], cone_k)
            if !Cones.use_scaling(cone_k)
                rhs4[z_rows_k] ./= solver.mu
            end
            @. @views rhs4[z_rows_k] += rhs[z_rows_k]
        else
            # -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
            @views Cones.hess_prod!(rhs4[z_rows_k], rhs[z_rows_k], cone_k)
            if !Cones.use_scaling(cone_k)
                rhs4[z_rows_k] .*= solver.mu
            end
            @. @views rhs4[z_rows_k] += rhs[s_rows_k]
        end
    end
    # -c'x - b'y - h'z + kapbar/taubar*tau = taurhs + kapbar/taubar*kaprhs
    kapontau = solver.kap / solver.tau
    rhs4[end] += kapontau * rhs[end]

    solve_subsystem(system_solver, sol4, rhs4)
    sol[1:dim4] .= sol4

    # lift to get s and kap
    # TODO refactor below for use with symindef and qrchol methods
    # s = -G*x + h*tau - zrhs
    s = @view sol[(dim4 + 1):(end - 1)]
    @. @views s = model.h * sol4[end] - rhs[(n + p) .+ (1:q)]
    @views mul!(s, model.G, sol[1:n], -1, true)

    # kap = kapbar/taubar*(kaprhs - tau)
    sol[end] = kapontau * (rhs[end] - sol4[end])

    return sol
end

#=
direct sparse
=#

mutable struct NaiveElimSparseSystemSolver{T <: Real} <: NaiveElimSystemSolver{T}
    use_inv_hess::Bool
    lhs4::SparseMatrixCSC # TODO CSC type will depend on factor cache Int type
    rhs4::Vector{T}
    sol4::Vector{T}
    hess_idxs::Vector
    fact_cache::SparseNonSymCache{T}
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
    cones = model.cones
    cone_idxs = model.cone_idxs

    system_solver.sol4 = zeros(T, n + p + q + 1)
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
    if isempty(cones)
        hess_nz_total = 0
    else
        hess_nz_total = sum(Cones.use_dual(cone_k) ? Cones.hess_nz_count(cone_k, false) : Cones.inv_hess_nz_count(cone_k, false) for cone_k in cones)
    end
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

function update_fact(system_solver::NaiveElimSparseSystemSolver, solver::Solver)
    for (k, cone_k) in enumerate(solver.model.cones)
        if Cones.use_dual(cone_k) # no scaling
            H = Cones.hess(cone_k)
            for j in 1:Cones.dimension(cone_k)
                nz_rows = Cones.hess_nz_idxs_col(cone_k, j, false)
                @. @views system_solver.lhs4.nzval[system_solver.hess_idxs[k][j]] = H[nz_rows, j] * solver.mu
            end
        else
            Hinv = Cones.inv_hess(cone_k)
            if Cones.use_scaling(cone_k)
                for j in 1:Cones.dimension(cone_k)
                    nz_rows = Cones.inv_hess_nz_idxs_col(cone_k, j, false)
                    @. @views system_solver.lhs4.nzval[system_solver.hess_idxs[k][j]] = Hinv[nz_rows, j]
                end
            else
                for j in 1:Cones.dimension(cone_k)
                    nz_rows = Cones.inv_hess_nz_idxs_col(cone_k, j, false)
                    @. @views system_solver.lhs4.nzval[system_solver.hess_idxs[k][j]] = Hinv[nz_rows, j] / solver.mu
                end
            end
        end
    end
    system_solver.lhs4.nzval[end] = solver.kap / solver.tau

    update_fact(system_solver.fact_cache, system_solver.lhs4)

    return system_solver
end

function solve_subsystem(system_solver::NaiveElimSparseSystemSolver, sol4::Vector, rhs4::Vector)
    solve_system(system_solver.fact_cache, sol4, system_solver.lhs4, rhs4)
    return sol4
end

#=
direct dense
=#

mutable struct NaiveElimDenseSystemSolver{T <: Real} <: NaiveElimSystemSolver{T}
    use_inv_hess::Bool
    lhs4::Matrix{T}
    rhs4::Vector{T}
    sol4::Vector{T}
    fact_cache::DenseNonSymCache{T}
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

    system_solver.sol4 = zeros(T, n + p + q + 1)
    system_solver.rhs4 = similar(system_solver.sol4)

    system_solver.lhs4 = T[
        zeros(T, n, n)  model.A'        model.G'                  model.c;
        -model.A        zeros(T, p, p)  zeros(T, p, q)            model.b;
        -model.G        zeros(T, q, p)  Matrix(one(T) * I, q, q)  model.h;
        -model.c'       -model.b'       -model.h'                 one(T);
        ]

    load_matrix(system_solver.fact_cache, system_solver.lhs4)

    return system_solver
end

function update_fact(system_solver::NaiveElimDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p) = (model.n, model.p)
    lhs4 = system_solver.lhs4

    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        z_rows_k = (n + p) .+ idxs_k
        if Cones.use_dual(cone_k) # no scaling
            # -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
            H = Cones.hess(cone_k)
            @. lhs4[z_rows_k, z_rows_k] = H * solver.mu
        elseif system_solver.use_inv_hess
            # -G_k*x + (mu*H_k)\z_k + h_k*tau = zrhs_k + (mu*H_k)\srhs_k
            Hinv = Cones.inv_hess(cone_k)
            if Cones.use_scaling(cone_k)
                @. lhs4[z_rows_k, z_rows_k] = Hinv
            else
                @. lhs4[z_rows_k, z_rows_k] = Hinv / solver.mu
            end
        else
            # -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
            @views Cones.hess_prod!(lhs4[z_rows_k, 1:n], model.G[idxs_k, :], cone_k)
            lhs4[z_rows_k, 1:n] .*= (Cones.use_scaling(cone_k) ? -1 : -solver.mu)
            @views Cones.hess_prod!(lhs4[z_rows_k, end], model.h[idxs_k], cone_k)
            if !Cones.use_scaling(cone_k)
                lhs4[z_rows_k, end] .*= solver.mu
            end
        end
    end
    lhs4[end, end] = solver.kap / solver.tau

    update_fact(system_solver.fact_cache, system_solver.lhs4)

    return system_solver
end

function solve_subsystem(system_solver::NaiveElimDenseSystemSolver, sol4::Vector, rhs4::Vector)
    copyto!(sol4, rhs4)
    solve_system(system_solver.fact_cache, sol4)
    # TODO recover if fails - check issuccess
    return sol4
end
