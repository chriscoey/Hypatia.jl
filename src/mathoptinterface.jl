#=
Copyright 2018, Chris Coey and contributors
=#

export WSOSPolyInterpCone

struct WSOSPolyInterpCone <: MOI.AbstractVectorSet
    dimension::Int
    ipwt::Vector{Matrix{Float64}}
    isdual::Bool
end
WSOSPolyInterpCone(dimension::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterpCone(dimension, ipwt, false)

mutable struct Optimizer <: MOI.AbstractOptimizer
    mdl::Model
    verbose::Bool
    timelimit::Float64
    lscachetype
    usedense::Bool
    P::AbstractMatrix{Float64}  # quadratic cost matrix, size n*n
    c::Vector{Float64}          # linear cost vector, size n
    A::AbstractMatrix{Float64}  # equality constraint matrix, size p*n
    b::Vector{Float64}          # equality constraint vector, size p
    G::AbstractMatrix{Float64}  # cone constraint matrix, size q*n
    h::Vector{Float64}          # cone constraint vector, size q
    cone::Cone                  # primal constraint cone object
    objsense::MOI.OptimizationSense
    objconst::Float64
    numeqconstrs::Int
    constroffseteq::Vector{Int}
    constrprimeq::Vector{Float64}
    constroffsetcone::Vector{Int}
    constrprimcone::Vector{Float64}
    intervalidxs
    intervalscales
    x::Vector{Float64}
    s::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    status::Symbol
    solvetime::Float64
    pobj::Float64
    dobj::Float64

    function Optimizer(mdl::Model, verbose::Bool, timelimit::Float64, lscachetype, usedense::Bool)
        opt = new()
        opt.mdl = mdl
        opt.verbose = verbose
        opt.timelimit = timelimit
        opt.lscachetype = lscachetype
        opt.usedense = usedense
        return opt
    end
end

Optimizer(;
    verbose::Bool = false,
    timelimit::Float64 = 3.6e3, # TODO should be Inf
    lscachetype = QRChol,
    usedense::Bool = true,
    tolrelopt::Float64 = 1e-6,
    tolabsopt::Float64 = 1e-7,
    tolfeas::Float64 = 1e-7,
    ) = Optimizer(Model(verbose=verbose, timelimit=timelimit, tolrelopt=tolrelopt, tolabsopt=tolabsopt, tolfeas=tolfeas), verbose, timelimit, lscachetype, usedense)

MOI.get(::Optimizer, ::MOI.SolverName) = "Hypatia"

MOI.is_empty(opt::Optimizer) = (get_status(opt.mdl) == :NotLoaded)
MOI.empty!(opt::Optimizer) = Optimizer() # TODO empty the data and results, or just create a new one? keep options?

MOI.supports(::Optimizer, ::Union{
    MOI.ObjectiveSense,
    MOI.ObjectiveFunction{MOI.SingleVariable},
    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
    MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}
    }) = true

# TODO don't restrict to Float64 type
SupportedFuns = Union{
    MOI.SingleVariable,
    MOI.ScalarAffineFunction{Float64},
    MOI.VectorOfVariables,
    MOI.VectorAffineFunction{Float64},
    }

SupportedSets = Union{
    MOI.EqualTo{Float64},
    MOI.Zeros,
    MOI.GreaterThan{Float64},
    MOI.Nonnegatives,
    MOI.LessThan{Float64},
    MOI.Nonpositives,
    MOI.Interval{Float64},
    MOI.SecondOrderCone,
    MOI.RotatedSecondOrderCone,
    MOI.ExponentialCone,
    MOI.GeometricMeanCone,
    MOI.PowerCone,
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.LogDetConeTriangle,
    WSOSPolyInterpCone,
    }

MOI.supports_constraint(::Optimizer, ::Type{<:SupportedFuns}, ::Type{<:SupportedSets}) = true

# MOI cones for which no transformation is needed
conefrommoi(s::MOI.SecondOrderCone) = EpiNormEucl(MOI.dimension(s))
conefrommoi(s::MOI.RotatedSecondOrderCone) = EpiPerSquare(MOI.dimension(s))
conefrommoi(s::MOI.ExponentialCone) = HypoPerLog()
conefrommoi(s::MOI.GeometricMeanCone) = (l = MOI.dimension(s) - 1; HypoGeomean(fill(1.0/l, l)))
conefrommoi(s::MOI.PowerCone) = EpiPerPower(inv(s.exponent))
conefrommoi(s::WSOSPolyInterpCone) = WSOSPolyInterp(s.dimension, s.ipwt, s.isdual)
conefrommoi(s::MOI.AbstractVectorSet) = error("MOI set $s is not recognized")

function buildvarcone(fi::MOI.VectorOfVariables, si::MOI.AbstractVectorSet, dim::Int, q::Int)
    IGi = q+1:q+dim
    VGi = -ones(dim)
    prmtvi = conefrommoi(si)
    return (IGi, VGi, prmtvi)
end

function buildconstrcone(fi::MOI.VectorAffineFunction{Float64}, si::MOI.AbstractVectorSet, dim::Int, q::Int)
    IGi = [q + vt.output_index for vt in fi.terms]
    VGi = [-vt.scalar_term.coefficient for vt in fi.terms]
    Ihi = q+1:q+dim
    Vhi = fi.constants
    prmtvi = conefrommoi(si)
    return (IGi, VGi, Ihi, Vhi, prmtvi)
end

# MOI cones requiring transformations (eg rescaling, changing order)
# TODO later remove if MOI gets scaled triangle sets
svecscale(dim) = [(i == j ? 1.0 : rt2) for i in 1:round(Int, sqrt(0.25 + 2*dim) - 0.5) for j in 1:i]
svecunscale(dim) = [(i == j ? 1.0 : rt2i) for i in 1:round(Int, sqrt(0.25 + 2*dim) - 0.5) for j in 1:i]

# PSD cone: convert from smat to svec form (scale off-diagonals)
function buildvarcone(fi::MOI.VectorOfVariables, si::MOI.PositiveSemidefiniteConeTriangle, dim::Int, q::Int)
    IGi = q+1:q+dim
    VGi = -svecscale(dim)
    prmtvi = PosSemidef(dim)
    return (IGi, VGi, prmtvi)
end

function buildconstrcone(fi::MOI.VectorAffineFunction{Float64}, si::MOI.PositiveSemidefiniteConeTriangle, dim::Int, q::Int)
    scalevec = svecscale(dim)
    IGi = [q + vt.output_index for vt in fi.terms]
    VGi = [-vt.scalar_term.coefficient*scalevec[vt.output_index] for vt in fi.terms]
    Ihi = q+1:q+dim
    Vhi = scalevec .* fi.constants
    prmtvi = PosSemidef(dim)
    return (IGi, VGi, Ihi, Vhi, prmtvi)
end

# logdet cone: convert from smat to svec form (scale off-diagonals)
function buildvarcone(fi::MOI.VectorOfVariables, si::MOI.LogDetConeTriangle, dim::Int, q::Int)
    IGi = q+1:q+dim
    VGi = vcat(-1.0, -1.0, -svecscale(dim-2))
    prmtvi = HypoPerLogdet(dim)
    return (IGi, VGi, prmtvi)
end

function buildconstrcone(fi::MOI.VectorAffineFunction{Float64}, si::MOI.LogDetConeTriangle, dim::Int, q::Int)
    scalevec = vcat(1.0, 1.0, svecscale(dim-2))
    IGi = [q + vt.output_index for vt in fi.terms]
    VGi = [-vt.scalar_term.coefficient*scalevec[vt.output_index] for vt in fi.terms]
    Ihi = q+1:q+dim
    Vhi = scalevec .* fi.constants
    prmtvi = HypoPerLogdet(dim)
    return (IGi, VGi, Ihi, Vhi, prmtvi)
end

# build representation as min c'x s.t. A*x = b, h - G*x in K
function MOI.copy_to(
    opt::Optimizer,
    src::MOI.ModelLike;
    copy_names::Bool = false,
    warn_attributes::Bool = true,
    )
    @assert !copy_names
    idxmap = Dict{MOI.Index, MOI.Index}()

    # variables
    n = MOI.get(src, MOI.NumberOfVariables()) # columns of A
    j = 0
    for vj in MOI.get(src, MOI.ListOfVariableIndices()) # MOI.VariableIndex
        j += 1
        idxmap[vj] = MOI.VariableIndex(j)
    end
    @assert j == n

    # objective function
    (IP, JP, VP) = (Int[], Int[], Float64[]) # quadratic terms
    (Jc, Vc) = (Int[], Float64[]) # affine terms

    F = MOI.get(src, MOI.ObjectiveFunctionType())
    obj = MOI.get(src, MOI.ObjectiveFunction{F}())
    if F == MOI.SingleVariable
        push!(Jc, idxmap[obj.variable].value)
        push!(Vc, 1.0)
    elseif F == MOI.ScalarAffineFunction{Float64}
        append!(Jc, idxmap[t.variable_index].value for t in obj.terms)
        append!(Vc, t.coefficient for t in obj.terms)
        opt.objconst = obj.constant
    elseif F == MOI.ScalarQuadraticFunction{Float64}
        append!(Jc, idxmap[t.variable_index].value for t in obj.affine_terms)
        append!(Vc, t.coefficient for t in obj.affine_terms)
        for t in obj.quadratic_terms # set upper triangle of P
            (i,j) = (idxmap[t.variable_index_1].value, idxmap[t.variable_index_2].value)
            if i > j
                (i,j) = (j,i)
            end
            push!(IP, i)
            push!(JP, j)
            push!(VP, t.coefficient)
        end
        opt.objconst = obj.constant
        @show opt.objconst
    end

    opt.objsense = MOI.get(src, MOI.ObjectiveSense())
    if opt.objsense == MOI.MaxSense
        Vc .= -Vc
        VP .= -VP
    end
    if opt.usedense
        opt.P = Symmetric(Matrix(sparse(IP, JP, VP, n, n)), :U)
    else
        opt.P = Symmetric(dropzeros!(sparse(IP, JP, VP, n, n)), :U)
    end
    opt.c = Vector(sparsevec(Jc, Vc, n))

    # constraints
    getsrccons(F, S) = MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
    getconfun(conidx) = MOI.get(src, MOI.ConstraintFunction(), conidx)
    getconset(conidx) = MOI.get(src, MOI.ConstraintSet(), conidx)
    i = 0 # MOI constraint index

    # equality constraints
    p = 0 # rows of A (equality constraint matrix)
    (IA, JA, VA) = (Int[], Int[], Float64[])
    (Ib, Vb) = (Int[], Float64[])
    (Icpe, Vcpe) = (Int[], Float64[]) # constraint set constants for opt.constrprimeq
    constroffseteq = Vector{Int}()

    # TODO can preprocess variables equal to constant
    for ci in getsrccons(MOI.SingleVariable, MOI.EqualTo{Float64})
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}(i)
        push!(constroffseteq, p)
        p += 1
        push!(IA, p)
        push!(JA, idxmap[getconfun(ci).variable].value)
        push!(VA, -1.0)
        push!(Ib, p)
        push!(Vb, -getconset(ci).value)
        push!(Icpe, p)
        push!(Vcpe, getconset(ci).value)
    end

    for ci in getsrccons(MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64})
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}(i)
        push!(constroffseteq, p)
        p += 1
        fi = getconfun(ci)
        for vt in fi.terms
            push!(IA, p)
            push!(JA, idxmap[vt.variable_index].value)
            push!(VA, -vt.coefficient)
        end
        push!(Ib, p)
        push!(Vb, fi.constant - getconset(ci).value)
        push!(Icpe, p)
        push!(Vcpe, getconset(ci).value)
    end

    # TODO can preprocess variables equal to zero here
    for ci in getsrccons(MOI.VectorOfVariables, MOI.Zeros)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Zeros}(i)
        push!(constroffseteq, p)
        fi = getconfun(ci)
        dim = MOI.output_dimension(fi)
        append!(IA, p+1:p+dim)
        append!(JA, idxmap[vi].value for vi in fi.variables)
        append!(VA, -ones(dim))
        p += dim
    end

    for ci in getsrccons(MOI.VectorAffineFunction{Float64}, MOI.Zeros)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros}(i)
        push!(constroffseteq, p)
        fi = getconfun(ci)
        dim = MOI.output_dimension(fi)
        for vt in fi.terms
            push!(IA, p + vt.output_index)
            push!(JA, idxmap[vt.scalar_term.variable_index].value)
            push!(VA, -vt.scalar_term.coefficient)
        end
        append!(Ib, p+1:p+dim)
        append!(Vb, fi.constants)
        p += dim
    end

    push!(constroffseteq, p)
    if opt.usedense
        opt.A = Matrix(sparse(IA, JA, VA, p, n))
    else
        opt.A = dropzeros!(sparse(IA, JA, VA, p, n))
    end
    opt.b = Vector(sparsevec(Ib, Vb, p)) # TODO if type less strongly, this can be sparse too
    opt.numeqconstrs = i
    opt.constrprimeq = Vector(sparsevec(Icpe, Vcpe, p))
    opt.constroffseteq = constroffseteq

    # conic constraints
    q = 0 # rows of G (cone constraint matrix)
    (IG, JG, VG) = (Int[], Int[], Float64[])
    (Ih, Vh) = (Int[], Float64[])
    (Icpc, Vcpc) = (Int[], Float64[]) # constraint set constants for opt.constrprimeq
    constroffsetcone = Vector{Int}()
    cone = Cone()

    # build up one nonnegative cone
    nonnegstart = q

    for ci in getsrccons(MOI.SingleVariable, MOI.GreaterThan{Float64})
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}(i)
        push!(constroffsetcone, q)
        q += 1
        push!(IG, q)
        push!(JG, idxmap[getconfun(ci).variable].value)
        push!(VG, -1.0)
        push!(Ih, q)
        push!(Vh, -getconset(ci).lower)
        push!(Vcpc, getconset(ci).lower)
        push!(Icpc, q)
    end

    for ci in getsrccons(MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64})
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}(i)
        push!(constroffsetcone, q)
        q += 1
        fi = getconfun(ci)
        for vt in fi.terms
            push!(IG, q)
            push!(JG, idxmap[vt.variable_index].value)
            push!(VG, -vt.coefficient)
        end
        push!(Ih, q)
        push!(Vh, fi.constant - getconset(ci).lower)
        push!(Vcpc, getconset(ci).lower)
        push!(Icpc, q)
    end

    for ci in getsrccons(MOI.VectorOfVariables, MOI.Nonnegatives)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Nonnegatives}(i)
        push!(constroffsetcone, q)
        for vj in getconfun(ci).variables
            q += 1
            push!(IG, q)
            push!(JG, idxmap[vj].value)
            push!(VG, -1.0)
        end
    end

    for ci in getsrccons(MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives}(i)
        push!(constroffsetcone, q)
        fi = getconfun(ci)
        dim = MOI.output_dimension(fi)
        for vt in fi.terms
            push!(IG, q + vt.output_index)
            push!(JG, idxmap[vt.scalar_term.variable_index].value)
            push!(VG, -vt.scalar_term.coefficient)
        end
        append!(Ih, q+1:q+dim)
        append!(Vh, fi.constants)
        q += dim
    end

    if q > nonnegstart
        # exists at least one nonnegative constraint
        addprimitivecone!(cone, Nonnegative(q - nonnegstart), nonnegstart+1:q)
    end

    # build up one nonpositive cone
    nonposstart = q

    for ci in getsrccons(MOI.SingleVariable, MOI.LessThan{Float64})
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}(i)
        push!(constroffsetcone, q)
        q += 1
        push!(IG, q)
        push!(JG, idxmap[getconfun(ci).variable].value)
        push!(VG, -1.0)
        push!(Ih, q)
        push!(Vh, -getconset(ci).upper)
        push!(Vcpc, getconset(ci).upper)
        push!(Icpc, q)
    end

    for ci in getsrccons(MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64})
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}(i)
        push!(constroffsetcone, q)
        q += 1
        fi = getconfun(ci)
        for vt in fi.terms
            push!(IG, q)
            push!(JG, idxmap[vt.variable_index].value)
            push!(VG, -vt.coefficient)
        end
        push!(Ih, q)
        push!(Vh, fi.constant - getconset(ci).upper)
        push!(Vcpc, getconset(ci).upper)
        push!(Icpc, q)
    end

    for ci in getsrccons(MOI.VectorOfVariables, MOI.Nonpositives)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Nonpositives}(i)
        push!(constroffsetcone, q)
        for vj in getconfun(ci).variables
            q += 1
            push!(IG, q)
            push!(JG, idxmap[vj].value)
            push!(VG, -1.0)
        end
    end

    for ci in getsrccons(MOI.VectorAffineFunction{Float64}, MOI.Nonpositives)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonpositives}(i)
        push!(constroffsetcone, q)
        fi = getconfun(ci)
        dim = MOI.output_dimension(fi)
        for vt in fi.terms
            push!(IG, q + vt.output_index)
            push!(JG, idxmap[vt.scalar_term.variable_index].value)
            push!(VG, -vt.scalar_term.coefficient)
        end
        append!(Ih, q+1:q+dim)
        append!(Vh, fi.constants)
        q += dim
    end

    if q > nonposstart
        # exists at least one nonpositive constraint
        addprimitivecone!(cone, Nonpositive(q - nonposstart), nonposstart+1:q)
    end

    # build up one L_infinity norm cone from two-sided interval constraints
    intervalstart = q
    nintervals = MOI.get(src, MOI.NumberOfConstraints{MOI.SingleVariable, MOI.Interval{Float64}}()) + MOI.get(src, MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64}, MOI.Interval{Float64}}())
    intervalscales = Vector{Float64}(undef, nintervals)

    if nintervals > 0
        i += 1
        push!(constroffsetcone, q)
        q += 1
        push!(Ih, q)
        push!(Vh, 1.0)
    end

    intervalcount = 0

    for ci in getsrccons(MOI.SingleVariable, MOI.Interval{Float64})
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.Interval{Float64}}(i)
        push!(constroffsetcone, q)
        q += 1

        upper = getconset(ci).upper
        lower = getconset(ci).lower
        @assert isfinite(upper) && isfinite(lower)
        @assert upper > lower
        mid = 0.5*(upper + lower)
        scal = 2.0*inv(upper - lower)

        push!(IG, q)
        push!(JG, idxmap[getconfun(ci).variable].value)
        push!(VG, -scal)
        push!(Ih, q)
        push!(Vh, -mid*scal)
        push!(Vcpc, mid)
        push!(Icpc, q)
        intervalcount += 1
        intervalscales[intervalcount] = scal
    end

    for ci in getsrccons(MOI.ScalarAffineFunction{Float64}, MOI.Interval{Float64})
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.Interval{Float64}}(i)
        push!(constroffsetcone, q)
        q += 1

        upper = getconset(ci).upper
        lower = getconset(ci).lower
        @assert isfinite(upper) && isfinite(lower)
        @assert upper > lower
        mid = 0.5*(upper + lower)
        scal = 2.0/(upper - lower)

        fi = getconfun(ci)
        for vt in fi.terms
            push!(IG, q)
            push!(JG, idxmap[vt.variable_index].value)
            push!(VG, -vt.coefficient*scal)
        end
        push!(Ih, q)
        push!(Vh, (fi.constant - mid)*scal)
        push!(Vcpc, mid)
        push!(Icpc, q)
        intervalcount += 1
        intervalscales[intervalcount] = scal
    end

    opt.intervalidxs = intervalstart+2:q
    opt.intervalscales = intervalscales
    if q > intervalstart
        # exists at least one interval-type constraint
        addprimitivecone!(cone, EpiNormInf(q - intervalstart), intervalstart+1:q)
    end

    # add non-LP conic constraints

    for S in (
        MOI.SecondOrderCone,
        MOI.RotatedSecondOrderCone,
        MOI.ExponentialCone,
        MOI.PowerCone,
        MOI.GeometricMeanCone,
        MOI.PositiveSemidefiniteConeTriangle,
        MOI.LogDetConeTriangle,
        WSOSPolyInterpCone,
        ),
        F in (MOI.VectorOfVariables, MOI.VectorAffineFunction{Float64})
        for ci in getsrccons(F, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{F, S}(i)
            push!(constroffsetcone, q)
            fi = getconfun(ci)
            si = getconset(ci)
            dim = MOI.output_dimension(fi)
            if F == MOI.VectorOfVariables
                append!(JG, idxmap[vj].value for vj in fi.variables)
                (IGi, VGi, prmtvi) = buildvarcone(fi, si, dim, q)
            else
                append!(JG, idxmap[vt.scalar_term.variable_index].value for vt in fi.terms)
                (IGi, VGi, Ihi, Vhi, prmtvi) = buildconstrcone(fi, si, dim, q)
                append!(Ih, Ihi)
                append!(Vh, Vhi)
            end
            append!(IG, IGi)
            append!(VG, VGi)
            addprimitivecone!(cone, prmtvi, q+1:q+dim)
            q += dim
        end
    end

    push!(constroffsetcone, q)
    if opt.usedense
        opt.G = Matrix(sparse(IG, JG, VG, q, n))
    else
        opt.G = dropzeros!(sparse(IG, JG, VG, q, n))
    end
    opt.h = Vector(sparsevec(Ih, Vh, q))
    opt.cone = cone
    opt.constroffsetcone = constroffsetcone
    opt.constrprimcone = Vector(sparsevec(Icpc, Vcpc, q))

    return idxmap
end

function MOI.optimize!(opt::Optimizer)
    mdl = opt.mdl
    (P, c, A, b, G, h, cone) = (opt.P, opt.c, opt.A, opt.b, opt.G, opt.h, opt.cone)

    # check, preprocess, load, and solve
    check_data(P, c, A, b, G, h, cone)
    # if opt.lscachetype == QRChol
    #     (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = preprocess_data(c, A, b, G, useQR=true)
    #     L = QRChol(c1, A1, b1, G1, h, cone, Q2, RiQ1)
    # elseif opt.lscachetype == Naive3
        # (P1, c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = preprocess_data(P, c, A, b, G, useQR=false)
        # L = Naive3(P1, c1, A1, b1, G1, h, cone)
        # L = Naive3(P, c, A, b, G, h, cone)
        L = Chol2(P, c, A, b, G, h, cone)
    # else
    #     error("linear system cache type $(opt.lscachetype) is not recognized")
    # end
    # load_data!(mdl, P1, c1, A1, b1, G1, h, cone, L)
    load_data!(mdl, P, c, A, b, G, h, cone, L)

    solve!(mdl)

    opt.status = get_status(mdl)
    opt.solvetime = get_solvetime(mdl)
    opt.pobj = get_pobj(mdl)
    opt.dobj = get_dobj(mdl)

    # get solution and transform for MOI
    opt.x = zeros(length(c))
    opt.x[dukeep] = get_x(mdl)
    opt.constrprimeq += opt.b - opt.A*opt.x
    opt.y = zeros(length(b))
    opt.y[prkeep] = get_y(mdl)

    opt.s = get_s(mdl)
    opt.z = get_z(mdl)

    for k in eachindex(cone.prmtvs)
        if cone.prmtvs[k] isa PosSemidef
            idxs = cone.idxs[k]
            scalevec = svecunscale(length(idxs))
            opt.s[idxs] .*= scalevec
            opt.z[idxs] .*= scalevec
        elseif cone.prmtvs[k] isa HypoPerLogdet
            idxs = cone.idxs[k][3:end]
            scalevec = svecunscale(length(idxs))
            opt.s[idxs] .*= scalevec
            opt.z[idxs] .*= scalevec
        end
    end

    opt.s[opt.intervalidxs] ./= opt.intervalscales
    opt.constrprimcone += opt.s
    opt.z[opt.intervalidxs] .*= opt.intervalscales

    return nothing
end

# function MOI.free!(opt::Optimizer) # TODO call gc on opt.mdl?

function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    # TODO time limit etc
    if opt.status in (:Optimal, :PrimalInfeasible, :DualInfeasible, :IllPosed)
        return MOI.Optimal
    elseif opt.status in (:PredictorFail, :CorrectorFail)
        return MOI.NumericalError
    elseif opt.status == :IterationLimit
        return MOI.IterationLimit
    elseif opt.status == :TimeLimit
        return MOI.TimeLimit
    else
        return MOI.OtherError
    end
end

function MOI.get(opt::Optimizer, ::MOI.ObjectiveValue)
    if opt.objsense == MOI.MinSense
        return opt.pobj + opt.objconst
    elseif opt.objsense == MOI.MaxSense
        return -opt.pobj + opt.objconst
    else
        error("no objective sense is set")
    end
end

function MOI.get(opt::Optimizer, ::MOI.ObjectiveBound)
    if opt.objsense == MOI.MinSense
        return opt.dobj + opt.objconst
    elseif opt.objsense == MOI.MaxSense
        return -opt.dobj + opt.objconst
    else
        error("no objective sense is set")
    end
end

function MOI.get(opt::Optimizer, ::MOI.ResultCount)
    if opt.status in (:Optimal, :PrimalInfeasible, :DualInfeasible, :IllPosed)
        return 1
    end
    return 0
end

function MOI.get(opt::Optimizer, ::MOI.PrimalStatus)
    if opt.status == :Optimal
        return MOI.FeasiblePoint
    elseif opt.status == :PrimalInfeasible
        return MOI.InfeasiblePoint
    elseif opt.status == :DualInfeasible
        return MOI.InfeasibilityCertificate
    elseif opt.status == :IllPosed
        return MOI.UnknownResultStatus # TODO later distinguish primal/dual ill posed certificates
    else
        return MOI.UnknownResultStatus
    end
end

function MOI.get(opt::Optimizer, ::MOI.DualStatus)
    if opt.status == :Optimal
        return MOI.FeasiblePoint
    elseif opt.status == :PrimalInfeasible
        return MOI.InfeasibilityCertificate
    elseif opt.status == :DualInfeasible
        return MOI.InfeasiblePoint
    elseif opt.status == :IllPosed
        return MOI.UnknownResultStatus # TODO later distinguish primal/dual ill posed certificates
    else
        return MOI.UnknownResultStatus
    end
end

MOI.get(opt::Optimizer, ::MOI.VariablePrimal, vi::MOI.VariableIndex) = opt.x[vi.value]
MOI.get(opt::Optimizer, a::MOI.VariablePrimal, vi::Vector{MOI.VariableIndex}) = MOI.get.(opt, a, vi)

function MOI.get(opt::Optimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractScalarSet}
    # scalar set
    i = ci.value
    if i <= opt.numeqconstrs
        # constraint is an equality
        return opt.y[opt.constroffseteq[i]+1]
    else
        # constraint is conic
        i -= opt.numeqconstrs
        return opt.z[opt.constroffsetcone[i]+1]
    end
end
function MOI.get(opt::Optimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractVectorSet}
    # vector set
    i = ci.value
    if i <= opt.numeqconstrs
        # constraint is an equality
        os = opt.constroffseteq
        return opt.y[os[i]+1:os[i+1]]
    else
        # constraint is conic
        i -= opt.numeqconstrs
        os = opt.constroffsetcone
        return opt.z[os[i]+1:os[i+1]]
    end
end
MOI.get(opt::Optimizer, a::MOI.ConstraintDual, ci::Vector{MOI.ConstraintIndex}) = MOI.get.(opt, a, ci)

function MOI.get(opt::Optimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractScalarSet}
    # scalar set
    i = ci.value
    if i <= opt.numeqconstrs
        # constraint is an equality
        return opt.constrprimeq[opt.constroffseteq[i]+1]
    else
        # constraint is conic
        i -= opt.numeqconstrs
        return opt.constrprimcone[opt.constroffsetcone[i]+1]
    end
end
function MOI.get(opt::Optimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractVectorSet}
    # vector set
    i = ci.value
    if i <= opt.numeqconstrs
        # constraint is an equality
        os = opt.constroffseteq
        return opt.constrprimeq[os[i]+1:os[i+1]]
    else
        # constraint is conic
        i -= opt.numeqconstrs
        os = opt.constroffsetcone
        return opt.constrprimcone[os[i]+1:os[i+1]]
    end
end
MOI.get(opt::Optimizer, a::MOI.ConstraintPrimal, ci::Vector{MOI.ConstraintIndex}) = MOI.get.(opt, a, ci)
