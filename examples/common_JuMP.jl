#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

common code for JuMP examples
=#

include(joinpath(@__DIR__, "common.jl"))

import JuMP
const MOI = JuMP.MOI

# SOCone, PSDCone, ExpCone, PowerCone only
MOI.Utilities.@model(ClassicConeOptimizer,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.SecondOrderCone, MOI.RotatedSecondOrderCone, MOI.PositiveSemidefiniteConeTriangle, MOI.ExponentialCone, MOI.DualExponentialCone,),
    (MOI.PowerCone, MOI.DualPowerCone,),
    (),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )

# SOCone only
MOI.Utilities.@model(SOConeOptimizer,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.SecondOrderCone, MOI.RotatedSecondOrderCone,),
    (),
    (),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )

# ExpCone only
MOI.Utilities.@model(ExpConeOptimizer,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.ExponentialCone,),
    (),
    (),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )

abstract type ExampleInstanceJuMP{T <: Real} <: ExampleInstance{T} end

# fallback: just check optimal status
function test_extra(inst::ExampleInstanceJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == MOI.OPTIMAL
end

# run a JuMP instance with a given solver and return solve info
function test(
    E::Type{<:ExampleInstanceJuMP{Float64}}, # an instance of a JuMP example # TODO support generic reals
    inst_data::Tuple,
    extender = nothing, # MOI.Utilities.@model-defined optimizer with subset of cones if using extended formulation
    solver_options = (), # additional non-default solver options specific to the example
    solver::Type{<:MOI.AbstractOptimizer} = Hypatia.Optimizer;
    default_solver_options = (), # default solver options
    process_extended_certificates::Bool = true, # whether to process the certificates for the extended space model (for Hypatia only) or the natural space model
    rseed::Int = 1,
    load_only::Bool = false,
    )
    # setup instance and model
    Random.seed!(rseed)
    inst = E(inst_data...)
    build_time = @elapsed model = build(inst)

    # solve
    opt = solver{Float64}(; default_solver_options..., solver_options...)
    if !isnothing(extender)
        # use MOI automated extended formulation
        opt = MOI.Bridges.full_bridge_optimizer(MOI.Utilities.CachingOptimizer(extender{Float64}(), opt), Float64)
    end
    JuMP.set_optimizer(model, () -> opt)
    JuMP.optimize!(model)

    # run tests for the example
    test_extra(inst, model)

    # process the solve info and solution
    if process_extended_certificates && solver <: Hypatia.Optimizer
        # use native process result function to calculate residuals on extended certificates stored inside the Hypatia optimizer struct
        hypatia_opt = get_inner_optimizer(model)
        result = process_result(hypatia_opt.model, hypatia_opt.solver)
    else
        result = process_result_JuMP(model)
    end

    return (extender, build_time, result)
end

# return solve information and certificate violations
# TODO finish natural space certificate checks and delete unused code
function process_result_JuMP(model::JuMP.Model)
    solve_time = JuMP.solve_time(model)
    num_iters = MOI.get(model, MOI.BarrierIterations())
    primal_obj = JuMP.objective_value(model)
    dual_obj = JuMP.dual_objective_value(model)

    # get data from extended Hypatia model
    hypatia_opt = get_inner_optimizer(model)
    hypatia_model = hypatia_opt.model
    (ext_n, ext_p, ext_q) = (hypatia_model.n, hypatia_model.p, hypatia_model.q)
    hypatia_status = Solvers.get_status(hypatia_opt.solver)
    # @show (ext_n, ext_p, ext_q)
    # @show typeof.(hypatia_model.cones)

    # get Hypatia native model in natural space from MOI.copy_to without extension
    model_backend = JuMP.backend(model)
    nat_hypatia_opt = Hypatia.Optimizer{Float64}()
    idx_map = MOI.copy_to(nat_hypatia_opt, model_backend)
    nat_hypatia_model = nat_hypatia_opt.model

    # get native certificates in natural space
    (nat_n, nat_p, nat_q) = (nat_hypatia_model.n, nat_hypatia_model.p, nat_hypatia_model.q)
    # @show (nat_n, nat_p, nat_q)
    # @show typeof.(nat_hypatia_model.cones)
    x = Vector{Float64}(undef, nat_n)
    y = Vector{Float64}(undef, nat_p)
    z = Vector{Float64}(undef, nat_q)
    s = similar(z)

    for (moi_idx, hyp_idx) in idx_map
        if moi_idx isa MOI.VariableIndex
            x[hyp_idx.value] = MOI.get(model_backend, MOI.VariablePrimal(), moi_idx)
        elseif moi_idx isa MOI.ConstraintIndex{<:MOI.AbstractScalarFunction, <:MOI.AbstractScalarSet}
            # moi_idx isa scalar set MOI.ConstraintIndex
            i = hyp_idx.value
            if i <= nat_hypatia_opt.num_eq_constrs
                # constraint is an equality
                K_idx = nat_hypatia_opt.constr_offset_eq[i] + 1
                y[K_idx] = MOI.get(model_backend, MOI.ConstraintDual(), moi_idx)
            else
                # constraint is conic - get primal and dual
                i -= nat_hypatia_opt.num_eq_constrs
                K_idx = nat_hypatia_opt.constr_offset_cone[i] + 1
                s[K_idx] = MOI.get(model_backend, MOI.ConstraintPrimal(), moi_idx)
                z[K_idx] = MOI.get(model_backend, MOI.ConstraintDual(), moi_idx)
            end
        else
            @assert moi_idx isa MOI.ConstraintIndex{<:MOI.AbstractVectorFunction, <:MOI.AbstractVectorSet}
            # moi_idx isa vector set MOI.ConstraintIndex
            i = hyp_idx.value
            if i <= nat_hypatia_opt.num_eq_constrs
                # constraint is an equality
                os = nat_hypatia_opt.constr_offset_eq
                K_idxs = (os[i] + 1):os[i + 1]
                y[K_idxs] = MOI.get(model_backend, MOI.ConstraintDual(), moi_idx)
            else
                # constraint is conic - get primal and dual
                os = nat_hypatia_opt.constr_offset_cone
                i -= nat_hypatia_opt.num_eq_constrs
                K_idxs = (os[i] + 1):os[i + 1]
                s[K_idxs] = MOI.get(model_backend, MOI.ConstraintPrimal(), moi_idx)
                z[K_idxs] = MOI.get(model_backend, MOI.ConstraintDual(), moi_idx)
            end
        end
    end

    # TODO transform z and s if necessary, or tell hypatia copy_to not to transform before
    # println()
    # @show x
    # @show y
    # @show z
    # @show s
    # println()

    # process certificates
    obj_diff = primal_obj - dual_obj
    compl = dot(s, z)
    (x_viol, y_viol, z_viol) = certificate_violations(hypatia_status, nat_hypatia_model, x, y, z, s)

    return (status = hypatia_status,
        solve_time = solve_time, num_iters = num_iters,
        primal_obj = primal_obj, dual_obj = dual_obj,
        n = ext_n, p = ext_p, q = ext_q,
        obj_diff = obj_diff, compl = compl,
        x_viol = x_viol, y_viol = y_viol, z_viol = z_viol)
end

function get_inner_optimizer(model::JuMP.Model)
    inner_model = JuMP.backend(model).optimizer.model
    if inner_model isa MOI.Bridges.LazyBridgeOptimizer
        return inner_model.model.optimizer
    else
        return inner_model.optimizer
    end
end
