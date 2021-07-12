#=
compute the capacity of a classical-quantum channel
adapted from https://github.com/hfawzi/cvxquad/blob/master/examples/cq_channel_capacity.m
and listing 1 in "Efficient optimization of the quantum relative entropy" by H. Fawzi and O. Fawzi
=#

struct ClassicalQuantum{T <: Real} <: ExampleInstanceJuMP{T}
    d::Int
    complex::Bool
    use_EF::Bool
end

function build(inst::ClassicalQuantum{T}) where {T <: Float64}
    d = inst.d
    @assert !(inst.complex && inst.use_EF)
    rt2 = sqrt(T(2))
    R = (inst.complex ? Complex{T} : T)

    function hermtr1()
        ρ = randn(R, d, d)
        ρ = ρ * ρ'
        ρ ./= tr(ρ)
        return Hermitian(ρ)
    end
    ρs = [hermtr1() for _ in 1:d]
    Hs = [dot(ρ, log(ρ)) for ρ in ρs]

    model = JuMP.Model()
    JuMP.@variable(model, prob[1:d] >= 0)
    JuMP.@constraint(model, sum(prob) == 1)
    JuMP.@variable(model, epi)
    JuMP.@objective(model, Max, -epi + dot(prob, Hs))

    entr = JuMP.AffExpr.(zeros(Cones.svec_length(R, d)))
    ρ_vec = zeros(T, length(entr))
    for (ρ, p) in zip(ρs, prob)
        Cones.smat_to_svec!(ρ_vec, ρ, rt2)
        JuMP.add_to_expression!.(entr, p, ρ_vec)
    end

    aff = vcat(epi, 1, entr)
    if inst.use_EF
        add_spectral(MatNegEntropyEigOrd(), d, aff, model)
    else
        JuMP.@constraint(model, aff in Hypatia.EpiPerSepSpectralCone{Float64}(
            Cones.NegEntropySSF(), Cones.MatrixCSqr{T, R}, d))
    end

    # save for use in tests
    model.ext[:epi] = epi
    model.ext[:entr] = entr

    return model
end

function test_extra(inst::ClassicalQuantum{T}, model::JuMP.Model) where T
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    (stat == MOI.OPTIMAL) || return

    # check constraint
    tol = eps(T)^0.2
    epi_opt = JuMP.value(model.ext[:epi])
    Entr_opt = zeros(inst.complex ? Complex{T} : T, inst.d, inst.d)
    Cones.svec_to_smat!(Entr_opt, JuMP.value.(model.ext[:entr]), sqrt(T(2)))
    λ = eigvals(Hermitian(Entr_opt, :U))
    @test minimum(λ) >= -tol
    qe_result = get_val(pos_only(λ), MatNegEntropy())
    @test epi_opt ≈ qe_result atol=tol rtol=tol
    return
end
