#=
Copyright 2018, Chris Coey and contributors

MathOptInterface wrapper of Hypatia solver
=#

mutable struct Optimizer{T <: Real} <: MOI.AbstractOptimizer
    load_only::Bool
    test_certificates::Bool

    solver::Solvers.Solver{T}
    model::Models.Model{T}

    result::NamedTuple
    x::Vector{T}
    s::Vector{T}
    y::Vector{T}
    z::Vector{T}

    obj_sense::MOI.OptimizationSense
    num_eq_constrs::Int
    constr_offset_eq::Vector{Int}
    constr_prim_eq::Vector{T}
    constr_offset_cone::Vector{Int}
    constr_prim_cone::Vector{T}
    interval_idxs::UnitRange{Int}
    interval_scales::Vector{T}

    function Optimizer{T}(;
        load_only::Bool = false,
        test_certificates::Bool = false,
        solver_options...
        ) where {T <: Real}
        opt = new{T}()
        opt.load_only = load_only
        opt.test_certificates = test_certificates
        opt.solver = Solvers.Solver{T}(; solver_options...)
        return opt
    end
end

Optimizer(; options...) = Optimizer{Float64}(; options...) # default to Float64

MOI.get(::Optimizer, ::MOI.SolverName) = "Hypatia"
MOI.get(opt::Optimizer, ::MOI.RawSolver) = opt.solver

MOI.is_empty(opt::Optimizer) = (opt.solver.status == :NotLoaded)

function MOI.empty!(opt::Optimizer)
    opt.solver.status = :NotLoaded
    opt.result = NamedTuple()
    return
end

MOI.supports(::Optimizer{T}, ::Union{
    MOI.ObjectiveSense,
    MOI.ObjectiveFunction{MOI.SingleVariable},
    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}},
    ) where {T <: Real} = true

MOI.supports_constraint(::Optimizer{T},
    ::Type{<:Union{MOI.SingleVariable, MOI.ScalarAffineFunction{T}}},
    ::Type{<:Union{MOI.EqualTo{T}, MOI.GreaterThan{T}, MOI.LessThan{T}, MOI.Interval{T}}}
    ) where {T <: Real} = true
MOI.supports_constraint(::Optimizer{T},
    ::Type{<:Union{MOI.VectorOfVariables, MOI.VectorAffineFunction{T}}},
    ::Type{<:Union{MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOIOtherCones{T}}}
    ) where {T <: Real} = true

# build representation as min c'x s.t. A*x = b, h - G*x in K
function MOI.copy_to(
    opt::Optimizer{T},
    src::MOI.ModelLike;
    copy_names::Bool = false,
    warn_attributes::Bool = true,
    ) where {T <: Real}
    @assert !copy_names
    idx_map = Dict{MOI.Index, MOI.Index}()

    # variables
    n = MOI.get(src, MOI.NumberOfVariables()) # columns of A
    j = 0
    for vj in MOI.get(src, MOI.ListOfVariableIndices()) # MOI.VariableIndex
        j += 1
        idx_map[vj] = MOI.VariableIndex(j)
    end
    @assert j == n

    # objective function
    F = MOI.get(src, MOI.ObjectiveFunctionType())
    if F == MOI.SingleVariable
        obj = MOI.ScalarAffineFunction{T}(MOI.get(src, MOI.ObjectiveFunction{F}()))
    elseif F == MOI.ScalarAffineFunction{T}
        obj = MOI.get(src, MOI.ObjectiveFunction{F}())
    end
    (Jc, Vc) = (Int[], T[])
    for t in obj.terms
        push!(Jc, idx_map[t.variable_index].value)
        push!(Vc, t.coefficient)
    end
    obj_offset = obj.constant
    if MOI.get(src, MOI.ObjectiveSense()) == MOI.MAX_SENSE
        Vc .*= -1
        obj_offset *= -1
    end
    opt.obj_sense = MOI.get(src, MOI.ObjectiveSense())
    model_c = Vector(sparsevec(Jc, Vc, n))

    # constraints
    get_src_cons(F, S) = MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
    get_con_fun(con_idx) = MOI.get(src, MOI.ConstraintFunction(), con_idx)
    get_con_set(con_idx) = MOI.get(src, MOI.ConstraintSet(), con_idx)
    i = 0 # MOI constraint index

    # equality constraints
    p = 0 # rows of A (equality constraint matrix)
    (IA, JA, VA) = (Int[], Int[], T[])
    (Ib, Vb) = (Int[], T[])
    (Icpe, Vcpe) = (Int[], T[]) # constraint set constants for opt.constr_prim_eq
    constr_offset_eq = Vector{Int}()

    for ci in get_src_cons(MOI.SingleVariable, MOI.EqualTo{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{T}}(i)
        push!(constr_offset_eq, p)
        p += 1
        push!(IA, p)
        push!(JA, idx_map[get_con_fun(ci).variable].value)
        push!(VA, -1)
        push!(Ib, p)
        push!(Vb, -get_con_set(ci).value)
        push!(Icpe, p)
        push!(Vcpe, get_con_set(ci).value)
    end

    for ci in get_src_cons(MOI.ScalarAffineFunction{T}, MOI.EqualTo{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}}(i)
        push!(constr_offset_eq, p)
        p += 1
        fi = get_con_fun(ci)
        for vt in fi.terms
            push!(IA, p)
            push!(JA, idx_map[vt.variable_index].value)
            push!(VA, -vt.coefficient)
        end
        push!(Ib, p)
        push!(Vb, fi.constant - get_con_set(ci).value)
        push!(Icpe, p)
        push!(Vcpe, get_con_set(ci).value)
    end

    for ci in get_src_cons(MOI.VectorOfVariables, MOI.Zeros)
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Zeros}(i)
        push!(constr_offset_eq, p)
        fi = get_con_fun(ci)
        dim = MOI.output_dimension(fi)
        append!(IA, (p + 1):(p + dim))
        append!(JA, idx_map[vi].value for vi in fi.variables)
        append!(VA, -ones(dim))
        p += dim
    end

    for ci in get_src_cons(MOI.VectorAffineFunction{T}, MOI.Zeros)
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{T}, MOI.Zeros}(i)
        push!(constr_offset_eq, p)
        fi = get_con_fun(ci)
        dim = MOI.output_dimension(fi)
        for vt in fi.terms
            push!(IA, p + vt.output_index)
            push!(JA, idx_map[vt.scalar_term.variable_index].value)
            push!(VA, -vt.scalar_term.coefficient)
        end
        append!(Ib, (p + 1):(p + dim))
        append!(Vb, fi.constants)
        p += dim
    end

    push!(constr_offset_eq, p)
    model_A = dropzeros!(sparse(IA, JA, VA, p, n))
    model_b = Vector(sparsevec(Ib, Vb, p))
    opt.num_eq_constrs = i
    opt.constr_prim_eq = Vector(sparsevec(Icpe, Vcpe, p))
    opt.constr_offset_eq = constr_offset_eq

    # conic constraints
    q = 0 # rows of G (cone constraint matrix)
    (IG, JG, VG) = (Int[], Int[], T[])
    (Ih, Vh) = (Int[], T[])
    (Icpc, Vcpc) = (Int[], T[]) # constraint set constants for opt.constr_prim_eq
    constr_offset_cone = Vector{Int}()
    cones = Cones.Cone{T}[]

    # build up one nonnegative cone
    nonneg_start = q

    for ci in get_src_cons(MOI.SingleVariable, MOI.GreaterThan{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{T}}(i)
        push!(constr_offset_cone, q)
        q += 1
        push!(IG, q)
        push!(JG, idx_map[get_con_fun(ci).variable].value)
        push!(VG, -1)
        push!(Ih, q)
        push!(Vh, -get_con_set(ci).lower)
        push!(Vcpc, get_con_set(ci).lower)
        push!(Icpc, q)
    end

    for ci in get_src_cons(MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T}}(i)
        push!(constr_offset_cone, q)
        q += 1
        fi = get_con_fun(ci)
        for vt in fi.terms
            push!(IG, q)
            push!(JG, idx_map[vt.variable_index].value)
            push!(VG, -vt.coefficient)
        end
        push!(Ih, q)
        push!(Vh, fi.constant - get_con_set(ci).lower)
        push!(Vcpc, get_con_set(ci).lower)
        push!(Icpc, q)
    end

    for ci in get_src_cons(MOI.VectorOfVariables, MOI.Nonnegatives)
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Nonnegatives}(i)
        push!(constr_offset_cone, q)
        for vj in get_con_fun(ci).variables
            q += 1
            push!(IG, q)
            push!(JG, idx_map[vj].value)
            push!(VG, -1)
        end
    end

    for ci in get_src_cons(MOI.VectorAffineFunction{T}, MOI.Nonnegatives)
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{T}, MOI.Nonnegatives}(i)
        push!(constr_offset_cone, q)
        fi = get_con_fun(ci)
        dim = MOI.output_dimension(fi)
        for vt in fi.terms
            push!(IG, q + vt.output_index)
            push!(JG, idx_map[vt.scalar_term.variable_index].value)
            push!(VG, -vt.scalar_term.coefficient)
        end
        append!(Ih, (q + 1):(q + dim))
        append!(Vh, fi.constants)
        q += dim
    end

    if q > nonneg_start
        # exists at least one nonnegative constraint
        push!(cones, Cones.Nonnegative{T}(q - nonneg_start))
    end

    # build up one nonpositive cone
    nonpos_start = q

    for ci in get_src_cons(MOI.SingleVariable, MOI.LessThan{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{T}}(i)
        push!(constr_offset_cone, q)
        q += 1
        push!(IG, q)
        push!(JG, idx_map[get_con_fun(ci).variable].value)
        push!(VG, -1)
        push!(Ih, q)
        push!(Vh, -get_con_set(ci).upper)
        push!(Vcpc, get_con_set(ci).upper)
        push!(Icpc, q)
    end

    for ci in get_src_cons(MOI.ScalarAffineFunction{T}, MOI.LessThan{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, MOI.LessThan{T}}(i)
        push!(constr_offset_cone, q)
        q += 1
        fi = get_con_fun(ci)
        for vt in fi.terms
            push!(IG, q)
            push!(JG, idx_map[vt.variable_index].value)
            push!(VG, -vt.coefficient)
        end
        push!(Ih, q)
        push!(Vh, fi.constant - get_con_set(ci).upper)
        push!(Vcpc, get_con_set(ci).upper)
        push!(Icpc, q)
    end

    for ci in get_src_cons(MOI.VectorOfVariables, MOI.Nonpositives)
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Nonpositives}(i)
        push!(constr_offset_cone, q)
        for vj in get_con_fun(ci).variables
            q += 1
            push!(IG, q)
            push!(JG, idx_map[vj].value)
            push!(VG, -1)
        end
    end

    for ci in get_src_cons(MOI.VectorAffineFunction{T}, MOI.Nonpositives)
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{T}, MOI.Nonpositives}(i)
        push!(constr_offset_cone, q)
        fi = get_con_fun(ci)
        dim = MOI.output_dimension(fi)
        for vt in fi.terms
            push!(IG, q + vt.output_index)
            push!(JG, idx_map[vt.scalar_term.variable_index].value)
            push!(VG, -vt.scalar_term.coefficient)
        end
        append!(Ih, (q + 1):(q + dim))
        append!(Vh, fi.constants)
        q += dim
    end

    if q > nonpos_start
        # exists at least one nonpositive constraint
        error("TODO fix now that Nonpositive cone is gone")
        push!(cones, Cones.Nonpositive{T}(q - nonpos_start))
    end

    # build up one L_infinity norm cone from two-sided interval constraints
    interval_start = q
    num_intervals = MOI.get(src, MOI.NumberOfConstraints{MOI.SingleVariable, MOI.Interval{T}}()) +
        MOI.get(src, MOI.NumberOfConstraints{MOI.ScalarAffineFunction{T}, MOI.Interval{T}}())
    interval_scales = Vector{T}(undef, num_intervals)

    if num_intervals > 0
        i += 1
        push!(constr_offset_cone, q)
        q += 1
        push!(Ih, q)
        push!(Vh, 1)
    end

    interval_count = 0

    for ci in get_src_cons(MOI.SingleVariable, MOI.Interval{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.Interval{T}}(i)
        push!(constr_offset_cone, q)
        q += 1

        upper = get_con_set(ci).upper
        lower = get_con_set(ci).lower
        @assert isfinite(upper) && isfinite(lower)
        @assert upper > lower
        mid = (upper + lower) / 2
        scal = 2 * inv(upper - lower)

        push!(IG, q)
        push!(JG, idx_map[get_con_fun(ci).variable].value)
        push!(VG, -scal)
        push!(Ih, q)
        push!(Vh, -mid * scal)
        push!(Vcpc, mid)
        push!(Icpc, q)
        interval_count += 1
        interval_scales[interval_count] = scal
    end

    for ci in get_src_cons(MOI.ScalarAffineFunction{T}, MOI.Interval{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, MOI.Interval{T}}(i)
        push!(constr_offset_cone, q)
        q += 1

        upper = get_con_set(ci).upper
        lower = get_con_set(ci).lower
        @assert isfinite(upper) && isfinite(lower)
        @assert upper > lower
        mid = (upper + lower) / 2
        scal = 2 / (upper - lower)

        fi = get_con_fun(ci)
        for vt in fi.terms
            push!(IG, q)
            push!(JG, idx_map[vt.variable_index].value)
            push!(VG, -vt.coefficient * scal)
        end
        push!(Ih, q)
        push!(Vh, (fi.constant - mid)*scal)
        push!(Vcpc, mid)
        push!(Icpc, q)
        interval_count += 1
        interval_scales[interval_count] = scal
    end

    opt.interval_idxs = (interval_start + 2):q
    opt.interval_scales = interval_scales
    if q > interval_start
        # exists at least one interval-type constraint
        push!(cones, Cones.EpiNormInf{T}(q - interval_start))
    end

    # add non-LP conic constraints

    for S in MOIOtherConesList(T), F in (MOI.VectorOfVariables, MOI.VectorAffineFunction{T})
        for ci in get_src_cons(F, S)
            i += 1
            idx_map[ci] = MOI.ConstraintIndex{F, S}(i)
            push!(constr_offset_cone, q)
            fi = get_con_fun(ci)
            si = get_con_set(ci)
            dim = MOI.output_dimension(fi)
            conei = cone_from_moi(T, si)
            if F == MOI.VectorOfVariables
                append!(JG, idx_map[vj].value for vj in fi.variables)
                IGi = (q + 1):(q + dim)
                VGi = -ones(T, dim)
            else
                append!(JG, idx_map[vt.scalar_term.variable_index].value for vt in fi.terms)
                IGi = [q + vt.output_index for vt in fi.terms]
                VGi = [-vt.scalar_term.coefficient for vt in fi.terms]
                Ihi = (q + 1):(q + dim)
                Vhi = fi.constants
                append!(Ih, Ihi)
                append!(Vh, Vhi)
            end
            append!(IG, IGi)
            append!(VG, VGi)
            push!(cones, conei)
            q += dim
        end
    end

    push!(constr_offset_cone, q)

    model_G = dropzeros!(sparse(IG, JG, VG, q, n))
    model_h = Vector(sparsevec(Ih, Vh, q))

    opt.model = Models.Model{T}(model_c, model_A, model_b, model_G, model_h, cones; obj_offset = obj_offset)

    opt.constr_offset_cone = constr_offset_cone
    opt.constr_prim_cone = Vector(sparsevec(Icpc, Vcpc, q))

    return idx_map
end

function MOI.optimize!(opt::Optimizer{T}) where {T <: Real}
    opt.load_only && return

    # build and solve the model
    model = opt.model
    opt.result = r = Solvers.solve_check(model, solver = opt.solver, test = opt.test_certificates)

    # transform solution for MOI conventions
    opt.x = r.x
    opt.constr_prim_eq += model.b - model.A * opt.x
    opt.y = r.y
    opt.s = r.s
    opt.z = r.z
    opt.s[opt.interval_idxs] ./= opt.interval_scales
    for (k, cone_k) in enumerate(model.cones)
        if cone_k isa Cones.PosSemidefTri || cone_k isa Cones.HypoPerLogdetTri # rescale duals for symmetric triangle cones
            cone_idxs_k = Models.get_cone_idxs(r.model)[k]
            unscale_vec = (Cones.use_dual(cone_k) ? opt.s : opt.z)
            idxs = (cone_k isa Cones.PosSemidefTri ? cone_idxs_k : cone_idxs_k[3:end])
            offset = 1
            for i in 1:round(Int, sqrt(0.25 + 2 * length(idxs)) - 0.5)
                for j in 1:(i - 1)
                    unscale_vec[idxs[offset]] /= 2
                    offset += 1
                end
                offset += 1
            end
        end
    end
    opt.constr_prim_cone += opt.s
    opt.z[opt.interval_idxs] .*= opt.interval_scales

    return
end

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.set(opt::Optimizer, ::MOI.Silent, value::Bool) = (opt.solver.verbose = value)
MOI.get(opt::Optimizer, ::MOI.Silent) = opt.solver.verbose

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.set(opt::Optimizer, ::MOI.TimeLimitSec, value::Real) = (opt.solver.time_limit = value)
MOI.set(opt::Optimizer, ::MOI.TimeLimitSec, ::Nothing) = (opt.solver.time_limit = Inf)
MOI.get(opt::Optimizer, ::MOI.TimeLimitSec) = (isfinite(opt.solver.time_limit) ? opt.solver.time_limit : nothing)

function MOI.get(opt::Optimizer, ::MOI.SolveTime)
    if opt.solver.status in (:NotLoaded, :Loaded)
        error("solve has not been called")
    end
    return opt.result.solve_time
end

MOI.get(opt::Optimizer, ::MOI.RawStatusString) = string(opt.solver.status)

function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    status = opt.solver.status
    if status in (:NotLoaded, :Loaded)
        return MOI.OPTIMIZE_NOT_CALLED
    elseif status == :Optimal
        return MOI.OPTIMAL
    elseif status == :PrimalInfeasible
        return MOI.INFEASIBLE
    elseif status == :DualInfeasible
        return MOI.DUAL_INFEASIBLE
    elseif status == :SlowProgress
        return MOI.SLOW_PROGRESS
    elseif status == :IterationLimit
        return MOI.ITERATION_LIMIT
    elseif status == :TimeLimit
        return MOI.TIME_LIMIT
    else
        @warn("Hypatia status $(opt.solver.status) not handled")
        return MOI.OTHER_ERROR
    end
end

function MOI.get(opt::Optimizer, ::MOI.PrimalStatus)
    status = opt.solver.status
    if status == :Optimal
        return MOI.FEASIBLE_POINT
    elseif status == :PrimalInfeasible
        return MOI.INFEASIBLE_POINT
    elseif status == :DualInfeasible
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif status == :IllPosed
        return MOI.OTHER_RESULT_STATUS
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(opt::Optimizer, ::MOI.DualStatus)
    status = opt.solver.status
    if status == :Optimal
        return MOI.FEASIBLE_POINT
    elseif status == :PrimalInfeasible
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif status == :DualInfeasible
        return MOI.INFEASIBLE_POINT
    elseif status == :IllPosed
        return MOI.OTHER_RESULT_STATUS
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(opt::Optimizer, ::MOI.ObjectiveValue)
    if opt.obj_sense == MOI.MIN_SENSE
        return opt.result.primal_obj
    elseif opt.obj_sense == MOI.MAX_SENSE
        return -opt.result.primal_obj
    else
        error("no objective sense is set")
    end
end

function MOI.get(opt::Optimizer, ::Union{MOI.DualObjectiveValue, MOI.ObjectiveBound})
    if opt.obj_sense == MOI.MIN_SENSE
        return opt.result.dual_obj
    elseif opt.obj_sense == MOI.MAX_SENSE
        return -opt.result.dual_obj
    else
        error("no objective sense is set")
    end
end

MOI.get(opt::Optimizer, ::MOI.ResultCount) = 1

MOI.get(opt::Optimizer, ::MOI.VariablePrimal, vi::MOI.VariableIndex) = opt.x[vi.value]
MOI.get(opt::Optimizer, a::MOI.VariablePrimal, vi::Vector{MOI.VariableIndex}) = MOI.get.(opt, a, vi)

function MOI.get(opt::Optimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractScalarSet}
    # scalar set
    i = ci.value
    if i <= opt.num_eq_constrs
        # constraint is an equality
        return opt.y[opt.constr_offset_eq[i] + 1]
    else
        # constraint is conic
        i -= opt.num_eq_constrs
        return opt.z[opt.constr_offset_cone[i] + 1]
    end
end
function MOI.get(opt::Optimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractVectorSet}
    # vector set
    i = ci.value
    if i <= opt.num_eq_constrs
        # constraint is an equality
        os = opt.constr_offset_eq
        return opt.y[(os[i] + 1):os[i + 1]]
    else
        # constraint is conic
        i -= opt.num_eq_constrs
        os = opt.constr_offset_cone
        return opt.z[(os[i] + 1):os[i + 1]]
    end
end
MOI.get(opt::Optimizer, a::MOI.ConstraintDual, ci::Vector{MOI.ConstraintIndex}) = MOI.get.(opt, a, ci)

function MOI.get(opt::Optimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractScalarSet}
    # scalar set
    i = ci.value
    if i <= opt.num_eq_constrs
        # constraint is an equality
        return opt.constr_prim_eq[opt.constr_offset_eq[i] + 1]
    else
        # constraint is conic
        i -= opt.num_eq_constrs
        return opt.constr_prim_cone[opt.constr_offset_cone[i] + 1]
    end
end
function MOI.get(opt::Optimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractVectorSet}
    # vector set
    i = ci.value
    if i <= opt.num_eq_constrs
        # constraint is an equality
        os = opt.constr_offset_eq
        return opt.constr_prim_eq[(os[i] + 1):os[i + 1]]
    else
        # constraint is conic
        i -= opt.num_eq_constrs
        os = opt.constr_offset_cone
        return opt.constr_prim_cone[(os[i] + 1):os[i + 1]]
    end
end
MOI.get(opt::Optimizer, a::MOI.ConstraintPrimal, ci::Vector{MOI.ConstraintIndex}) = MOI.get.(opt, a, ci)
