#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

signomial minimization problem
see "Signomial and Polynomial Optimization via Relative Entropy and Partial Dualization" (2019) by Murray, Chandrasekaran, & Wierman

a signomial f = Sig(A, c) of variables
    x in R^n
with parameters
    c in R^m
    A in R^{m, n}
takes values
    f(x) = sum_{i in [m]} c_i exp(A_{i, :} * x)

for signomials f, (g_p)_{p in 1:q}, the signomial minimization problem is
    inf_{x in R^n} f(x) : g_p(x) >= 0, p in 1:q

the convex SAGE relaxation problem is
    max_{γ in R, μ in R_+^q} γ : c - γ - sum_{p in 1:q} μ_p * g_p in C_SAGE(A)

a vector d in R^m belongs to SAGE cone C_SAGE(A) if
    exists C in R^{m, m} :
    d = sum_{k in [m]} C_{k, :}
    C_{k, :} in C_AGE(A, k)
which is equivalent to the feasibility problem over C in R^{m, m} and V in R^{m, m-1}
    d = sum_{k in [m]} C_{k, :}
    for k in [m]:
        (A_{\k, :} - [1,...,1] * A_{k, :}')' * V_{k, :} == [0,...,0]
        [C_{k, k} + sum(V_{k, :}), C_{k, \k}, V_{k, :}] in RelEntr(1 + 2(m - 1))
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))
include(joinpath(@__DIR__, "data.jl"))

struct SignomialMinJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    fc::Vector
    fA::AbstractMatrix
    gc::Vector
    gA::Vector
    obj_ub::Real
end
SignomialMinJuMP{Float64}(sig_name::Symbol) = SignomialMinJuMP{Float64}(signomialmin_data[sig_name]...)
SignomialMinJuMP{Float64}(m::Int, n::Int) = SignomialMinJuMP{Float64}(signomialmin_random(m, n)...)

example_tests(::Type{SignomialMinJuMP{Float64}}, ::MinimalInstances) = [
    ((:CS16ex12,),),
    ((2, 2),),
    ((2, 2), ClassicConeOptimizer),
    ]
example_tests(::Type{SignomialMinJuMP{Float64}}, ::FastInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    relaxed_options = (tol_feas = 1e-5, tol_rel_opt = 1e-4, tol_abs_opt = 1e-4)
    return [
    ((:motzkin2,), nothing, options),
    ((:motzkin2,), ClassicConeOptimizer, options),
    ((:motzkin3,), nothing, options),
    ((:CS16ex8_13,), nothing, options),
    ((:CS16ex8_14,), nothing, options),
    ((:CS16ex18,), nothing, options),
    ((:CS16ex12,), nothing, options),
    ((:CS16ex13,), nothing, options),
    ((:MCW19ex1_mod,), nothing, options),
    ((:MCW19ex8,), nothing, relaxed_options),
    ((:MCW19ex8,), ClassicConeOptimizer, relaxed_options),
    ((3, 2), nothing, options),
    ((3, 2), ClassicConeOptimizer, options),
    ((6, 6), nothing, options),
    ((20, 3), nothing, options),
    ((20, 3), ClassicConeOptimizer, options),
    ]
end
example_tests(::Type{SignomialMinJuMP{Float64}}, ::SlowInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    relaxed_options = (tol_feas = 1e-5, tol_rel_opt = 1e-4, tol_abs_opt = 1e-4)
    return [
    ((10, 10), nothing, options),
    ((10, 10), ClassicConeOptimizer, options),
    ((20, 6), nothing, options),
    ((40, 3), nothing, options),
    ]
end
example_tests(::Type{SignomialMinJuMP{Float64}}, ::ExpInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    relaxed_options = (tol_feas = 1e-5, tol_rel_opt = 1e-4, tol_abs_opt = 1e-4)
    return [
    ((:motzkin2,), ClassicConeOptimizer, options),
    # ((:MCW19ex8,), ClassicConeOptimizer, relaxed_options),
    # ((3, 2), ClassicConeOptimizer, options),
    # ((20, 3), ClassicConeOptimizer, options),
    # ((10, 10), ClassicConeOptimizer, options),
    ]
end

function build(inst::SignomialMinJuMP{T}) where {T <: Float64} # TODO generic reals
    (fc, fA, gc, gA) = (inst.fc, inst.fA, inst.gc, inst.gA)
    (fm, n) = size(fA)
    @assert length(fc) == fm
    q = length(gc)
    @assert length(gA) == q
    @assert all(size(gA_p, 2) == n for gA_p in gA)
    @assert all(size(gA_p, 1) == length(gc_p) for (gc_p, gA_p) in zip(gc, gA))

    # find unique terms
    unique_terms = Dict{Vector, Tuple{Vector{Int}, Vector{Tuple{Int, Int}}}}()
    for k in 1:size(fA, 1)
        row_k = fA[k, :]
        if row_k in keys(unique_terms)
            push!(unique_terms[row_k][1], k)
        else
            unique_terms[row_k] = (Int[k], Tuple{Int, Int}[])
        end
    end
    for (p, gA_p) in enumerate(gA)
        for k in 1:size(gA_p, 1)
            p_row_k = gA_p[k, :]
            if p_row_k in keys(unique_terms)
                push!(unique_terms[p_row_k][2], (p, k))
            else
                unique_terms[p_row_k] = (Int[], Tuple{Int, Int}[(p, k)])
            end
        end
    end
    A = vcat((row_k' for row_k in keys(unique_terms))...)
    m = size(A, 1)

    model = JuMP.Model()
    JuMP.@variable(model, γ)
    JuMP.@objective(model, Max, γ)
    JuMP.@variable(model, μ[1:q] >= 0)

    d = zeros(JuMP.AffExpr, m)
    const_found = false
    for k in 1:m
        row_k = A[k, :]
        (ks, pks) = unique_terms[row_k]
        for k2 in ks
            d[k] += fc[k2]
        end
        for (p, k2) in pks
            d[k] -= μ[p] * gc[p][k2]
        end
        if iszero(norm(row_k))
            # row is a constant term
            d[k] -= γ
            @assert !const_found
            const_found = true
        end
    end
    if !const_found
        A = vcat(A, zeros(A, 1, n))
        d = vcat(d, -γ)
    end

    # setup SAGE constraints
    notk = [[l for l in 1:m if l != k] for k in 1:m]
    JuMP.@variable(model, C[1:m, 1:m])
    JuMP.@variable(model, V[1:m, 1:(m - 1)])
    JuMP.@constraint(model, [k in 1:m], d[k] == sum(C[:, k]))
    JuMP.@constraint(model, [k in 1:m, i in 1:n], dot(A[notk[k], i] .- A[k, i], V[k, :]) == 0)
    JuMP.@constraint(model, [k in 1:m], vcat(C[k, k] + sum(V[k, :]), C[k, notk[k]], V[k, :]) in MOI.RelativeEntropyCone(2m - 1))

    return model
end

function test_extra(inst::SignomialMinJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    if JuMP.termination_status(model) == MOI.OPTIMAL && !isnan(inst.obj_ub)
        # check objective value is correct
        tol = eps(T)^0.2
        @test JuMP.objective_value(model) <= inst.obj_ub + tol
    end
end

return SignomialMinJuMP
