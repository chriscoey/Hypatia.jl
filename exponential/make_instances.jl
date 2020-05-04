# #
# using Hypatia
#
#
# function juliaize_instance(
#     E::Type{<:ExampleInstanceJuMP{Float64}},
#     inst_data::Tuple,
#     extender = nothing;
#     rseed::Int = 1
#     )
#     Random.seed!(rseed)
#     inst = E(inst_data...)
#     model = build(inst)
#     hyp_opt = Hypatia.Optimizer()
#     set_optimizer(model, () -> hyp_opt)
#
#     if load_only
#         MOIU.attach_optimizer(backend(model))
#         cc_opt = backend(model).optimizer.model
#         hyp_opt = Hypatia.Optimizer(; solver_options...)
#         MOI.copy_to(hyp_opt, cc_opt)
#         JuMP.set_optimizer(model, () -> hyp_opt)
#     end
#
# end
