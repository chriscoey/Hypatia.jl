#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

symmetric-indefinite linear system solver
solves linear system in naive.jl by first eliminating s and kap via the method in naiveelim.jl and then eliminating tau via a procedure similar to that described by S7.4 of
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf




TODO maybe move the c, b, h rhs div to the update fact part




3x3 nonsymmetric system in (x, y, z):
A'*y + G'*z = [xrhs, -c]
-A*x = [yrhs, -b]
(pr bar) -mu*H_k*G_k*x + z_k = [mu*H_k*zrhs_k + srhs_k, -mu*H_k*h_k]
(du bar) -G_k*x + mu*H_k*z_k = [zrhs_k + srhs_k, -h_k]

multiply pr bar constraint by (mu*H_k)^-1 to get 3x3 symmetric indefinite system
A'*y + G'*z = [xrhs, -c]
A*x = [-yrhs, b]
(pr bar) G_k*x - (mu*H_k)\z_k = [-zrhs_k - (mu*H_k)\srhs_k, h_k]
(du bar) G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]

or to avoid inverse hessian products, let for pr bar w_k = (mu*H_k)\z_k (later recover z_k = mu*H_k*w_k) to get 3x3 symmetric indefinite system
A'*y + sum_{pr bar} G_k'*mu*H_k*w_k + sum_{du bar} G_k'*z_k = [xrhs, -c]
A*x = [-yrhs, b]
(pr bar) mu*H_k*G_k*x - mu*H_k*w_k = [-mu*H_k*zrhs_k - srhs_k, mu*H_k*h_k]
(du bar) G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]
=#

abstract type SymIndefSystemSolver{T <: Real} <: SystemSolver{T} end

# TODO refac to use this for QRChol too
function solve_system(system_solver::SymIndefSystemSolver{T}, solver::Solver{T}, sol::Vector{T}, rhs::Vector{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)

    solve_subsystem() # TODO

    # TODO below could be shorter - don't need to actually separate out the x,y,z pieces - just use the constant rhs3 dot product
    sol_const = system_solver.sol_const
    x_const = @view sol_const[1:n]
    y_const = @view sol_const[n .+ (1:p)]
    z_const = @view sol_const[(n + p) .+ (1:q)]
    x_new = @view sol3[1:n]
    y_new = @view sol3[n .+ (1:p)]
    z_new = @view sol3[(n + p) .+ (1:q)]

    # lift to get tau
    # TODO maybe use higher precision here
    tau_denom = solver.kap / solver.tau - dot(model.c, x_const) - dot(model.b, y_const) - dot(model.h, z_const) # TODO store once
    tau = @view sol[dim3:dim3, :]
    @. @views tau = rhs[dim3:dim3, :] + rhs[end:end, :]
    tau .+= model.c' * x_new + model.b' * y_new + model.h' * z_new # TODO in place
    @. tau /= tau_denom

    @. x_new += tau * x_const
    @. y_new += tau * y_const
    @. z_new += tau * z_const

    @views sol[1:(dim3 - 1), :] = sol3[:]

    # lift to get s and kap
    # TODO refactor below for use with symindef and qrchol methods
    # s = -G*x + h*tau - zrhs
    s = @view sol[(dim3 + 1):(end - 1), :]
    mul!(s, model.h, tau)
    mul!(s, model.G, sol[1:n, :], -one(T), true)
    @. @views s -= rhs[(n + p) .+ (1:q), :]

    # kap = -mu/(taubar^2)*tau + kaprhs
    @. @views sol[end:end, :] = -solver.mu / solver.tau * tau / solver.tau + rhs[end:end, :]

    return sol
end

function solve_subsystem()
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)

    sol3 = system_solver.sol3
    rhs3 = system_solver.rhs3
    dim3 = size(sol3, 1)

    @. @views rhs3[1:n, 1] = rhs[1:n]
    @. @views rhs3[n .+ (1:p), 1] = -rhs[n .+ (1:p)]

    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        z_rows_k = (n + p) .+ idxs_k
        s_rows_k = dim3 .+ idxs_k
        zk12 = @view rhs[z_rows_k, :]
        sk12 = @view rhs[s_rows_k, :]
        hk = @view model.h[idxs_k]
        zk12_new = @view rhs3[z_rows_k]
        zk3_new = @view rhs3[z_rows_k, 3]

        if Cones.use_dual(cone_k)
            # G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]
            @. zk12_new = -zk12 - sk12
            @. zk3_new = hk
        elseif system_solver.use_inv_hess
            # G_k*x - (mu*H_k)\z_k = [-zrhs_k - (mu*H_k)\srhs_k, h_k]
            Cones.inv_hess_prod!(zk12_new, sk12, cone_k)
            @. zk12_new *= -1
            @. zk12_new -= zk12
            @. zk3_new = hk
        else
            # A'*y + sum_{pr bar} G_k'*mu*H_k*w_k + sum_{du bar} G_k'*z_k = [xrhs, -c]
            # mu*H_k*G_k*x - mu*H_k*w_k = [-mu*H_k*zrhs_k - srhs_k, mu*H_k*h_k]
            Cones.hess_prod!(zk12_new, zk12, cone_k)
            @. zk12_new *= -1
            @. zk12_new -= sk12
            Cones.hess_prod!(zk3_new, hk, cone_k)
        end
    end

    solve_subsubsystem(system_solver, sol3, rhs3)

    if !system_solver.use_inv_hess
        for (k, cone_k) in enumerate(model.cones)
            if !Cones.use_dual(cone_k)
                # recover z_k = mu*H_k*w_k
                z_rows_k = (n + p) .+ model.cone_idxs[k]
                z_copy_k = sol3[z_rows_k, :] # TODO do in-place
                @views Cones.hess_prod!(sol3[z_rows_k, :], z_copy_k, cone_k)
            end
        end
    end

    return sol
end

#=
direct sparse
=#

mutable struct SymIndefSparseSystemSolver{T <: Real} <: SymIndefSystemSolver{T}
    use_inv_hess::Bool
    lhs3::SparseMatrixCSC # TODO type will depend on Int type
    rhs3::Vector{T}
    sol3::Vector{T}
    hess_idxs::Vector
    fact_cache::SparseSymCache{T}
    function SymIndefSparseSystemSolver{Float64}(;
        use_inv_hess::Bool = true,
        fact_cache::SparseSymCache{Float64} = SparseSymCache{Float64}(),
        )
        system_solver = new{Float64}()
        if !use_inv_hess
            @warn("SymIndefSparseSystemSolver is not implemented with `use_inv_hess` set to `false`, using `true` instead.")
        end
        system_solver.use_inv_hess = true
        system_solver.fact_cache = fact_cache
        return system_solver
    end
end

function load(system_solver::SymIndefSparseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.fact_cache.analyzed = false
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs

    system_solver.sol3 = zeros(n + p + q, 2)
    system_solver.rhs3 = similar(system_solver.sol3)

    # form sparse LHS without Hessians and inverse Hessians in z/z block
    lhs3 = T[
        spzeros(T, n, n)  spzeros(T, n, p)  spzeros(T, n, q);
        model.A           spzeros(T, p, p)  spzeros(T, p, q);
        model.G           spzeros(T, q, p)  sparse(-one(T) * I, q, q);
        ]
    @assert issparse(lhs3)
    dropzeros!(lhs3)
    (Is, Js, Vs) = findnz(lhs3)

    # add I, J, V for Hessians and inverse Hessians
    if isempty(cones)
        hess_nz_total = 0
    else
        hess_nz_total = sum(Cones.use_dual(cone_k) ? Cones.hess_nz_count(cone_k, true) : Cones.inv_hess_nz_count(cone_k, true) for cone_k in cones)
    end
    H_Is = Vector{Int}(undef, hess_nz_total)
    H_Js = Vector{Int}(undef, hess_nz_total)
    offset = 1
    y_start = n + p - 1
    for (k, cone_k) in enumerate(cones)
        cone_idxs_k = cone_idxs[k]
        z_start_k = y_start + first(cone_idxs_k)
        for j in 1:Cones.dimension(cone_k)
            nz_rows_kj = z_start_k .+ (Cones.use_dual(cone_k) ? Cones.hess_nz_idxs_col(cone_k, j, true) : Cones.inv_hess_nz_idxs_col(cone_k, j, true))
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

    pert = T(system_solver.fact_cache.diag_pert) # TODO not happy with where this is stored
    append!(Is, 1:(n + p))
    append!(Js, 1:(n + p))
    append!(Vs, fill(pert, n))
    append!(Vs, fill(-pert, p))

    dim = size(lhs3, 1)
    # integer type supported by the sparse system solver library to be used
    Ti = int_type(system_solver.fact_cache)
    # prefer conversions of integer types to happen here than inside external wrappers
    Is = convert(Vector{Ti}, Is)
    Js = convert(Vector{Ti}, Js)
    lhs3 = system_solver.lhs3 = sparse(Is, Js, Vs, dim, dim)

    # cache indices of nonzeros of Hessians and inverse Hessians in sparse LHS nonzeros vector
    system_solver.hess_idxs = [Vector{Union{UnitRange, Vector{Int}}}(undef, Cones.dimension(cone_k)) for cone_k in cones]
    for (k, cone_k) in enumerate(cones)
        cone_idxs_k = cone_idxs[k]
        z_start_k = y_start + first(cone_idxs_k)
        for j in 1:Cones.dimension(cone_k)
            col = z_start_k + j
            # get nonzero rows in the current column of the LHS
            col_idx_start = lhs3.colptr[col]
            nz_rows = lhs3.rowval[col_idx_start:(lhs3.colptr[col + 1] - 1)]
            # get nonzero rows in column j of the Hessian or inverse Hessian
            nz_hess_indices = (Cones.use_dual(cone_k) ? Cones.hess_nz_idxs_col(cone_k, j, true) : Cones.inv_hess_nz_idxs_col(cone_k, j, true))
            # get index corresponding to first nonzero Hessian element of the current column of the LHS
            first_H = findfirst(isequal(z_start_k + first(nz_hess_indices)), nz_rows)
            # indices of nonzero values for cone k column j
            system_solver.hess_idxs[k][j] = (col_idx_start + first_H - 2) .+ (1:length(nz_hess_indices))
        end
    end

    return system_solver
end

function update_fact(system_solver::SymIndefSparseSystemSolver, solver::Solver)
    for (k, cone_k) in enumerate(solver.model.cones)
        H = (Cones.use_dual(cone_k) ? Cones.hess(cone_k) : Cones.inv_hess(cone_k))
        for j in 1:Cones.dimension(cone_k)
            nz_rows = (Cones.use_dual(cone_k) ? Cones.hess_nz_idxs_col(cone_k, j, true) : Cones.inv_hess_nz_idxs_col(cone_k, j, true))
            @. @views system_solver.lhs3.nzval[system_solver.hess_idxs[k][j]] = -H[nz_rows, j]
        end
    end

    update_fact(system_solver.fact_cache, system_solver.lhs3)

    return system_solver
end

function solve_subsubsystem(system_solver::SymIndefSparseSystemSolver, sol3::Vector, rhs3::Vector)
    solve_system(system_solver.fact_cache, sol3, system_solver.lhs3, rhs3)
    return sol3
end

#=
direct dense
=#

mutable struct SymIndefDenseSystemSolver{T <: Real} <: SymIndefSystemSolver{T}
    use_inv_hess::Bool
    lhs3::Symmetric{T, Matrix{T}}
    rhs3::Vector{T}
    sol3::Vector{T}
    fact_cache::DenseSymCache{T}
    function SymIndefDenseSystemSolver{T}(;
        use_inv_hess::Bool = true,
        fact_cache::DenseSymCache{T} = DenseSymCache{T}(),
        ) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_inv_hess = use_inv_hess
        system_solver.fact_cache = fact_cache
        return system_solver
    end
end

function load(system_solver::SymIndefDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)

    system_solver.sol3 = zeros(T, n + p + q)
    system_solver.rhs3 = similar(system_solver.sol3)

    # fill symmetric lower triangle
    system_solver.lhs3 = Symmetric(T[
        zeros(T, n, n)  zeros(T, n, p)  zeros(T, n, q);
        model.A         zeros(T, p, p)  zeros(T, p, q);
        model.G         zeros(T, q, p)  Matrix(-one(T) * I, q, q);
        ], :L)

    load_matrix(system_solver.fact_cache, system_solver.lhs3)

    return system_solver
end

function update_fact(system_solver::SymIndefDenseSystemSolver, solver::Solver)
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    lhs3 = system_solver.lhs3.data

    # update 3x3 LHS matrix
    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        z_rows_k = (n + p) .+ idxs_k
        if Cones.use_dual(cone_k)
            # G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]
            H = Cones.hess(cone_k)
            @. lhs3[z_rows_k, z_rows_k] = -H
        elseif system_solver.use_inv_hess
            # G_k*x - (mu*H_k)\z_k = [-zrhs_k - (mu*H_k)\srhs_k, h_k]
            Hinv = Cones.inv_hess(cone_k)
            @. lhs3[z_rows_k, z_rows_k] = -Hinv
        else
            # A'*y + sum_{pr bar} G_k'*mu*H_k*w_k + sum_{du bar} G_k'*z_k = [xrhs, -c]
            # mu*H_k*G_k*x - mu*H_k*w_k = [-mu*H_k*zrhs_k - srhs_k, mu*H_k*h_k]
            H = Cones.hess(cone_k)
            @. lhs3[z_rows_k, z_rows_k] = -H
            @views Cones.hess_prod!(lhs3[z_rows_k, 1:n], model.G[idxs_k, :], cone_k)
        end
    end

    update_fact(system_solver.fact_cache, system_solver.lhs3)

    # solve for constant RHS with b, c, h
    @. rhs3[1:n] = -model.c
    @. rhs3[1:n] = model.b
    @. rhs3[(n + p) .+ (1:q)] = model.h

    sol3 = system_solver.sol3
    solve_subsystem(system_solver, sol3, rhs3)




    return system_solver
end

function solve_subsubsystem(system_solver::SymIndefDenseSystemSolver, sol3::Vector, rhs3::Vector)
    copyto!(sol3, rhs3)
    solve_system(system_solver.fact_cache, sol3)
    # TODO recover if fails - check issuccess
    return sol3
end
