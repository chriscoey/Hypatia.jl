#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

common code for JuMP examples
=#

include(joinpath(@__DIR__, "common.jl"))

import JuMP
const MOI = JuMP.MOI

# SOCone, PSDCone, ExpCone, PowerCone only
MOI.Utilities.@model(StandardConeOptimizer,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.SecondOrderCone, MOI.RotatedSecondOrderCone, MOI.PositiveSemidefiniteConeTriangle, MOI.ExponentialCone,),
    (MOI.PowerCone, MOI.DualPowerCone,),
    (),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )

# SOCone and PSDCone only
MOI.Utilities.@model(SOPSDConeOptimizer,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.SecondOrderCone, MOI.RotatedSecondOrderCone,
    MOI.PositiveSemidefiniteConeTriangle,),
    (),
    (),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )

# ExpCone and PSDCone only
MOI.Utilities.@model(ExpPSDConeOptimizer,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.ExponentialCone, MOI.PositiveSemidefiniteConeTriangle,),
    (),
    (),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )

abstract type ExampleInstanceJuMP{T <: Real} <: ExampleInstance{T} end

# fallback: just check optimal status
function test_extra(inst::ExampleInstanceJuMP{Float64}, model::JuMP.Model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
end

function run_instance(
    ex_type::Type{<:ExampleInstanceJuMP{Float64}}, # an instance of a JuMP example
    inst_data::Tuple,
    extender = nothing, # MOI.Utilities.@model-defined optimizer with subset of cones if using extended formulation
    inst_options::NamedTuple = NamedTuple(),
    solver_type = Hypatia.Optimizer;
    default_options::NamedTuple = NamedTuple(),
    test::Bool = true,
    )
    new_options = merge(default_options, inst_options)

    println("setup model")
    setup_time = @elapsed (model, model_stats) = setup_model(ex_type, inst_data, extender, new_options, solver_type)

    println("solve and check")
    check_time = @elapsed solve_stats = solve_check(model, test = test)

    return (model_stats..., solve_stats..., setup_time, check_time)
end

function setup_model(
    ex_type::Type{<:ExampleInstanceJuMP{Float64}},
    inst_data::Tuple,
    extender,
    solver_options::NamedTuple,
    solver_type;
    rseed::Int = 1,
    )
    # setup example instance and JuMP model
    Random.seed!(rseed)
    inst = ex_type(inst_data...)
    model = build(inst)

    opt = hyp_opt = (solver_type == Hypatia.Optimizer) ? Hypatia.Optimizer(; solver_options...) : Hypatia.Optimizer()
    if !isnothing(extender)
        # use MOI automated extended formulation
        opt = MOI.Bridges.full_bridge_optimizer(MOI.Utilities.CachingOptimizer(extender{Float64}(), opt), Float64)
    end
    backend = JuMP.backend(model)
    MOI.Utilities.reset_optimizer(backend, opt)
    MOI.Utilities.attach_optimizer(backend)
    isnothing(extender) || MOI.Utilities.attach_optimizer(backend.optimizer.model)
    flush(stdout); flush(stderr)

    hyp_model = hyp_opt.model
    if solver_type != Hypatia.Optimizer
        # not using Hypatia to solve, so setup new JuMP model corresponding to Hypatia data
        (A, b, c, G, h) = (hyp_model.A, hyp_model.b, hyp_model.c, hyp_model.G, hyp_model.h)
        (cones, cone_idxs) = (hyp_model.cones, hyp_model.cone_idxs)

        new_model = JuMP.Model()
        new_model.ext[:hyp_data] = hyp_model
        JuMP.@variable(new_model, x_var[1:length(c)])
        JuMP.@objective(new_model, Min, dot(c, x_var))
        eq_refs = JuMP.@constraint(new_model, A * x_var .== b)
        cone_refs = Vector{JuMP.ConstraintRef}(undef, length(cones))
        for (k, cone_k) in enumerate(cones)
            idxs = cone_idxs[k]
            h_k = h[idxs]
            G_k = G[idxs, :]
            moi_set = cone_from_hyp(cone_k)
            if Hypatia.needs_untransform(moi_set)
                Hypatia.untransform_affine(moi_set, h_k)
                for j in 1:size(G_k, 2)
                    @views Hypatia.untransform_affine(moi_set, G_k[:, j])
                end
            end
            cone_refs[k] = JuMP.@constraint(new_model, h_k - G_k * x_var in moi_set)
        end
        new_model.ext[:x_var] = x_var
        new_model.ext[:eq_refs] = eq_refs
        new_model.ext[:cone_refs] = cone_refs

        opt = solver_type(; solver_options...)
        JuMP.set_optimizer(new_model, () -> opt)
        model = new_model
        flush(stdout); flush(stderr)
    else
        model.ext[:inst] = inst
    end

    string_cones = [string(nameof(c)) for c in unique(typeof.(hyp_model.cones))]
    model_stats = (hyp_model.n, hyp_model.p, hyp_model.q, string_cones)
    return (model, model_stats)
end

function solve_check(
    model::JuMP.Model;
    test::Bool = true,
    )
    JuMP.optimize!(model) # TODO make sure it doesn't copy again - just the optimize call - maybe use MOI.optimize instead
    # MOI.optimize!(opt)
    flush(stdout); flush(stderr)

    opt = JuMP.backend(model).optimizer
    if !isa(opt, Hypatia.Optimizer)
        backend_model = JuMP.backend(model).optimizer.model
        if backend_model isa MOI.Utilities.CachingOptimizer
            opt = backend_model.optimizer
        end
    end

    if opt isa Hypatia.Optimizer
        test && test_extra(model.ext[:inst], model)
        flush(stdout); flush(stderr)
        return process_result(opt.model, opt.solver)
    end

    test && @info("cannot run example tests if solver is not Hypatia")

    solve_time = JuMP.solve_time(model)
    num_iters = MOI.get(model, MOI.BarrierIterations())
    primal_obj = JuMP.objective_value(model)
    dual_obj = JuMP.dual_objective_value(model)
    moi_status = MOI.get(model, MOI.TerminationStatus())
    hyp_status = haskey(moi_hyp_status_map, moi_status) ? moi_hyp_status_map[moi_status] : :OtherStatus

    hyp_data = model.ext[:hyp_data]
    eq_refs = model.ext[:eq_refs]
    cone_refs = model.ext[:cone_refs]
    x = JuMP.value.(model.ext[:x_var])
    y = (isempty(eq_refs) ? Float64[] : -JuMP.dual.(eq_refs))
    s_cones = Vector{Vector{Float64}}(undef, length(cone_refs))
    z_cones = Vector{Vector{Float64}}(undef, length(cone_refs))
    for (k, cr) in enumerate(cone_refs)
        moi_set = MOI.get(cr.model, MOI.ConstraintSet(), cr)
        idxs = Hypatia.permute_affine(moi_set, 1:length(hyp_data.cone_idxs[k]))
        s_k = Hypatia.rescale_affine(moi_set, JuMP.value.(cr))
        z_k = Hypatia.rescale_affine(moi_set, JuMP.dual.(cr))
        s_cones[k] = s_k[idxs]
        z_cones[k] = z_k[idxs]
    end
    s = vcat(s_cones...)
    z = vcat(z_cones...)

    obj_diff = primal_obj - dual_obj
    compl = dot(s, z)
    (x_viol, y_viol, z_viol) = certificate_violations(hyp_status, hyp_data, x, y, z, s)
    flush(stdout); flush(stderr)

    solve_stats = (hyp_status, solve_time, num_iters, primal_obj, dual_obj, obj_diff, compl, x_viol, y_viol, z_viol)
    return solve_stats
end

# run a CBF instance with a given solver and return solve info
function test(
    inst::String, # a CBF file name
    solver_options = (), # additional non-default solver options specific to the example
    solver_type = Hypatia.Optimizer,
    )
    cbf_file = joinpath(cblib_dir, inst * ".cbf.gz")
    model = JuMP.read_from_file(cbf_file)

    # delete integer constraints
    int_cons = JuMP.all_constraints(model, JuMP.VariableRef, MOI.Integer)
    JuMP.delete.(model, int_cons)

    opt = solver_type(; solver_options...)
    JuMP.set_optimizer(model, () -> opt)
    JuMP.optimize!(model)
    flush(stdout); flush(stderr)

    @test JuMP.termination_status(model) == MOI.OPTIMAL # TODO some may be infeasible

    return process_result(model)
end

moi_hyp_status_map = Dict(
    MOI.OPTIMAL => :Optimal,
    MOI.INFEASIBLE => :PrimalInfeasible,
    MOI.DUAL_INFEASIBLE => :DualInfeasible,
    )

cone_from_hyp(cone::Cones.Cone) = error("cannot transform a Hypatia cone of type $(typeof(cone)) to an MOI cone")
cone_from_hyp(cone::Cones.Nonnegative) = MOI.Nonnegatives(Cones.dimension(cone))
cone_from_hyp(cone::Cones.EpiNormInf) = (Cones.use_dual_barrier(cone) ? MOI.NormOneCone : MOI.NormInfinityCone)(Cones.dimension(cone))
cone_from_hyp(cone::Cones.EpiNormEucl) = MOI.SecondOrderCone(Cones.dimension(cone))
cone_from_hyp(cone::Cones.EpiPerSquare) = MOI.RotatedSecondOrderCone(Cones.dimension(cone))
cone_from_hyp(cone::Cones.HypoPerLog) = (@assert Cones.dimension(cone) == 3; MOI.ExponentialCone())
cone_from_hyp(cone::Cones.EpiSumPerEntropy) = MOI.RelativeEntropyCone(Cones.dimension(cone))
cone_from_hyp(cone::Cones.HypoGeoMean) = MOI.GeometricMeanCone(Cones.dimension(cone))
cone_from_hyp(cone::Cones.Power) = (@assert Cones.dimension(cone) == 3; MOI.PowerCone{Float64}(cone.alpha[1]))
cone_from_hyp(cone::Cones.EpiNormSpectral) = (Cones.use_dual_barrier(cone) ? MOI.NormNuclearCone : MOI.NormSpectralCone)(cone.n, cone.m)
cone_from_hyp(cone::Cones.PosSemidefTri{T, R}) where {R <: Hypatia.RealOrComplex{T}} where {T <: Real} = MOI.PositiveSemidefiniteConeTriangle(cone.side)
cone_from_hyp(cone::Cones.LinMatrixIneq{T}) where {T <: Real} = Hypatia.LinMatrixIneqCone{T}(cone.As)
cone_from_hyp(cone::Cones.HypoPerLogdetTri) = MOI.LogDetConeTriangle(cone.side)
cone_from_hyp(cone::Cones.HypoRootdetTri) = MOI.RootDetConeTriangle(cone.side)
cone_from_hyp(cone::Cones.MatrixEpiPerSquare{T, R}) where {R <: Hypatia.RealOrComplex{T}} where {T <: Real} = Hypatia.MatrixEpiPerSquareCone{T, R}(cone.n, cone.m)
