#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

function mpbcones_to_hypatiacones!(
    hypatia_cone::Hypatia.Cone,
    mpb_cones::Vector{Tuple{Symbol,Vector{Int}}},
    parametric_refs::Vector{Int},
    parameters::Vector{Vector{Float64}},
    offset::Int=0,
    )
    power_cones_count = 0
    for (i, c) in enumerate(mpb_cones)
        (cone_symbol, idx_list) = (c[1], c[2])
        if cone_symbol in (:Zero, :Free)
            continue
        end

        smallest_ind = minimum(idx_list)
        start_ind = offset + 1
        end_ind = offset + maximum(idx_list) - minimum(idx_list) + 1
        output_idxs = UnitRange{Int}(start_ind, end_ind)
        offset += length(c[2])

        if cone_symbol == :Power
            power_cones_count += 1
            alphas = parameters[parametric_refs[power_cones_count]]
            hypatia_alpha = sum(alphas) / alphas[1]
            prmtv = Hypatia.EpiPerPower(hypatia_alpha, false)
        elseif cone_symbol == :NonPos
            prmtv = Hypatia.Nonpositive(length(output_idxs), false)
        elseif cone_symbol == :NonNeg
            prmtv = Hypatia.Nonnegative(length(output_idxs), false)
        elseif cone_symbol == :SOC
            prmtv = Hypatia.EpiNormEucl(length(output_idxs), false)
        elseif cone_symbol == :SOCRotated
            prmtv = Hypatia.EpiPerSquare(length(output_idxs), false)
        elseif cone_symbol == :ExpPrimal
            prmtv = Hypatia.HypoPerLog(false)
        elseif cone_symbol == :SDP
            prmtv = Hypatia.PosSemidef(length(output_idxs), false)
        end

        push!(hypatia_cone.prmtvs, prmtv)
        push!(hypatia_cone.idxs, output_idxs)
    end

    return hypatia_cone
end

function mpbtohypatia(
    c_in::Vector{Float64},
    A_in::AbstractMatrix,
    b_in::Vector{Float64},
    con_cones::Vector{Tuple{Symbol,Vector{Int}}},
    var_cones::Vector{Tuple{Symbol,Vector{Int}}},
    sense::Symbol,
    con_power_refs::Vector{Int},
    var_power_refs::Vector{Int},
    power_alphas::Vector{Vector{Float64}},
    objoffset::Float64,
    usedense::Bool,
    )
    # dimension of x
    n = length(c_in)

    for p in power_alphas
        if length(p) > 2
            error("we cannot convert to a power cone with more than three variables yet")
        else
            @assert sum(p) ≈ 1.0
        end
    end

    # count the number of "zero" constraints
    zero_constrs_count = 0
    cone_constrs_count = 0
    for (cone_type, inds) in con_cones
        if cone_type == :Zero
            zero_constrs_count += length(inds)
        else
            cone_constrs_count += length(inds)
        end
    end

    # count the number of cone variables
    cone_vars_count = 0
    zero_vars = 0
    cone_var_inds = Int[]
    zero_var_inds = Int[]
    zero_var_cones = Int[]
    cone_count = 0
    for (cone_type, inds) in var_cones
        cone_count += 1
        if cone_type == :Zero
            push!(zero_var_inds, inds...)
            push!(zero_var_cones, cone_count)
            zero_vars += length(inds)
        elseif cone_type != :Free
            cone_vars_count += length(inds)
            push!(cone_var_inds, inds...)
        end
    end
    @assert length(cone_var_inds) == cone_vars_count
    # variables that are fixed at zero count as constraints
    zero_constrs_count += zero_vars

    h = zeros(cone_constrs_count + cone_vars_count)
    b = zeros(zero_constrs_count)
    if usedense
        A = zeros(zero_constrs_count, n)
        G = zeros(cone_constrs_count + cone_vars_count, n)
    else
        A = spzeros(zero_constrs_count, n)
        G = spzeros(cone_constrs_count + cone_vars_count, n)
    end

    # keep index of constraints in A and G
    i = 0
    j = 0
    # constraints are split among A and G
    for (cone_type, inds) in con_cones
        if cone_type == :Zero
            nexti = i + length(inds)
            out_inds = i+1:nexti
            A[out_inds, :] = A_in[inds, :]
            b[out_inds] = b_in[inds]
            i = nexti
        else
            nextj = j + length(inds)
            out_inds = j+1:nextj
            G[out_inds, :] = A_in[inds, :]
            h[out_inds] = b_in[inds]
            j = nextj
        end
    end
    # corner case, add variables fixed at zero as constraints
    if zero_vars > 0
        fixed_var_ref = zero_constrs_count-zero_vars+1:zero_constrs_count
        @assert all(b[fixed_var_ref] .≈ 0.0)
        @assert all(A[fixed_var_ref, zero_var_inds] .≈ 0.0)
        @assert length(zero_var_inds) == zero_vars
        for ind in zero_var_inds
            i += 1
            A[i, ind] = 1.0
        end
    end

    # append G
    G[cone_constrs_count+1:end, :] = Matrix(-1.0I, n, n)[cone_var_inds, :]

    # prepare Hypatia cone
    hypatia_cone = Hypatia.Cone()
    mpbcones_to_hypatiacones!(hypatia_cone, con_cones, con_power_refs, power_alphas)
    mpbcones_to_hypatiacones!(hypatia_cone, var_cones, var_power_refs, power_alphas, cone_constrs_count)

    return (c_in, A, b, G, h, hypatia_cone)
end

function cbftohypatia(dat::CBFData; remove_ints::Bool=false, usedense::Bool=false)
    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat, col_major=true, roundints=true)
    if dat.sense == :Max
        c .*= -1.0
    end
    if remove_ints
        (c, A, b, con_cones, var_cones, vartypes) = remove_ints_in_nonlinear_cones(c, A, b, con_cones, var_cones, vartypes)
    end

    (c, A, b, G, h, hypatia_cone) = mpbtohypatia(c, A, b, con_cones, var_cones, dat.sense, dat.con_power_refs, dat.var_power_refs, dat.power_cone_alphas, dat.objoffset, usedense)
    hasintegervars = !isempty(dat.intlist)

    return (c, A, b, G, h, hypatia_cone, dat.objoffset, hasintegervars)
end
