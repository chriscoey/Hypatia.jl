
using JuMP, ECOS

T = Float64

tol = sqrt(sqrt(eps(T)))
Trt2 = sqrt(T(2))
Tirt2 = inv(Trt2)

model = Model(with_optimizer(ECOS.Optimizer))
@variable(model, x[1:3])
@objective(model, Min, -x[2] - x[3])
@constraints(model, begin
    x1, x[1] == 1
    x2, x[2] == Tirt2
    co, x in SecondOrderCone()
end)

optimize!(model)

@show value.(x)
@show dual(x1)
@show dual(x2)
@show dual(co)
@show objective_value(model)

# if you put the following code in Solvers.jl right before starting IPM main loop, then Hypatia only takes 4 iterations
# println()
# @show point.x
# @show point.y
# @show point.z
# @show point.s
#
# point.x .= [1.1, 1/sqrt(2), 1/sqrt(2)]
# point.y .= [sqrt(2), 0]
# point.z .= [sqrt(2) + 0.1, -1, -1]
# point.s .= point.x
# println()
#
# @show point.x
# @show point.y
# @show point.z
# @show point.s
# println()
