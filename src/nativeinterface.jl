#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz
=#

# model object containing options, problem data, linear system cache, and solution
mutable struct Model
    # options
    verbose::Bool           # if true, prints progress at each iteration
    timelimit::Float64      # (approximate) time limit (in seconds) for algorithm in solve function
    tolrelopt::Float64      # relative optimality gap tolerance
    tolabsopt::Float64      # absolute optimality gap tolerance
    tolfeas::Float64        # feasibility tolerance
    maxiter::Int            # maximum number of iterations
    predlinesearch::Bool    # if false, predictor step uses a fixed step size, else step size is determined via line search
    maxpredsmallsteps::Int  # maximum number of predictor step size reductions allowed with respect to the safe fixed step size
    predlsmulti::Float64    # predictor line search step size multiplier
    corrcheck::Bool         # if false, maxcorrsteps corrector steps are performed at each corrector phase, else the corrector phase can be terminated before maxcorrsteps corrector steps if the iterate is in the eta-neighborhood
    maxcorrsteps::Int       # maximum number of corrector steps (possible values: 1, 2, or 4)
    alphacorr::Float64      # corrector step size
    maxcorrlsiters::Int     # maximum number of line search iterations in each corrector step
    corrlsmulti::Float64    # corrector line search step size multiplier

    # problem data
    P::AbstractMatrix{Float64}  # quadratic cost matrix, size n*n
    c::Vector{Float64}          # linear cost vector, size n
    A::AbstractMatrix{Float64}  # equality constraint matrix, size p*n
    b::Vector{Float64}          # equality constraint vector, size p
    G::AbstractMatrix{Float64}  # cone constraint matrix, size q*n
    h::Vector{Float64}          # cone constraint vector, size q
    cone::Cone                  # primal constraint cone object

    L::LinSysCache  # cache for linear system solves

    # results
    status::Symbol          # solver status
    solvetime::Float64      # total solve time
    niters::Int             # total number of iterations

    pobj::Float64           # final primal objective value
    dobj::Float64           # final dual objective value
    x::Vector{Float64}      # final value of the primal free variables
    s::Vector{Float64}      # final value of the primal cone variables
    y::Vector{Float64}      # final value of the dual free variables
    z::Vector{Float64}      # final value of the dual cone variables
    mu::Float64             # final value of mu

    function Model(verbose, timelimit, tolrelopt, tolabsopt, tolfeas, maxiter, predlinesearch, maxpredsmallsteps, predlsmulti, corrcheck, maxcorrsteps, alphacorr, maxcorrlsiters, corrlsmulti)
        mdl = new()
        mdl.verbose = verbose
        mdl.timelimit = timelimit
        mdl.tolrelopt = tolrelopt
        mdl.tolabsopt = tolabsopt
        mdl.tolfeas = tolfeas
        mdl.maxiter = maxiter
        mdl.predlinesearch = predlinesearch
        mdl.maxpredsmallsteps = maxpredsmallsteps
        mdl.predlsmulti = predlsmulti
        mdl.corrcheck = corrcheck
        mdl.maxcorrsteps = maxcorrsteps
        mdl.alphacorr = alphacorr
        mdl.maxcorrlsiters = maxcorrlsiters
        mdl.corrlsmulti = corrlsmulti
        mdl.status = :NotLoaded
        return mdl
    end
end

# initialize a model object
function Model(;
    verbose = false,
    timelimit = 3.6e3, # TODO should be Inf
    tolrelopt = 1e-6,
    tolabsopt = 1e-7,
    tolfeas = 1e-7,
    maxiter = 1e4,
    predlinesearch = true,
    maxpredsmallsteps = 15,
    predlsmulti = 0.7,
    corrcheck = true,
    maxcorrsteps = 15,
    alphacorr = 1.0,
    maxcorrlsiters = 15,
    corrlsmulti = 0.5,
    )
    if min(tolrelopt, tolabsopt, tolfeas) < 1e-12 || max(tolrelopt, tolabsopt, tolfeas) > 1e-2
        error("tolrelopt, tolabsopt, tolfeas must be between 1e-12 and 1e-2")
    end
    if timelimit < 1e-2
        error("timelimit must be at least 1e-2")
    end
    if maxiter < 1
        error("maxiter must be at least 1")
    end
    if maxpredsmallsteps < 1
        error("maxpredsmallsteps must be at least 1")
    end
    if maxcorrsteps < 1
        error("maxcorrsteps must be at least 1")
    end

    return Model(verbose, timelimit, tolrelopt, tolabsopt, tolfeas, maxiter, predlinesearch, maxpredsmallsteps, predlsmulti, corrcheck, maxcorrsteps, alphacorr, maxcorrlsiters, corrlsmulti)
end

get_status(mdl::Model) = mdl.status
get_solvetime(mdl::Model) = mdl.solvetime
get_niters(mdl::Model) = mdl.niters
get_pobj(mdl::Model) = mdl.pobj
get_dobj(mdl::Model) = mdl.dobj
get_x(mdl::Model) = copy(mdl.x)
get_s(mdl::Model) = copy(mdl.s)
get_y(mdl::Model) = copy(mdl.y)
get_z(mdl::Model) = copy(mdl.z)
get_mu(mdl::Model) = mdl.mu


# check data for consistency
function check_data(
    P::AbstractMatrix{Float64},
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    cone::Cone,
    )
    (n, p, q) = (length(c), length(b), length(h))
    # if iszero(P)
    #     error("P matrix is zero, so perhaps you meant to use the cone LP algorithm")
    # end
    if n == 0
        error("c vector is empty, but number of primal variables must be positive")
    end
    if q == 0
        println("no conic constraints were specified; proceeding anyway")
    end
    if n < p
        println("number of equality constraints ($p) exceeds number of variables ($n)")
    end
    if n != size(P, 1) || n != size(P, 2) || !issymmetric(P)
        error("P is not a symmetric matrix with dimensions equal to number of primal variables")
    end
    if n != size(A, 2) || n != size(G, 2)
        error("number of variables is not consistent in A, G, and c")
    end
    if p != size(A, 1)
        error("number of constraint rows is not consistent in A and b")
    end
    if q != size(G, 1)
        error("number of constraint rows is not consistent in G and h")
    end

    if length(cone.prmtvs) != length(cone.idxs)
        error("number of primitive cones does not match number of index ranges")
    end
    qcone = 0
    for k in eachindex(cone.prmtvs)
        if dimension(cone.prmtvs[k]) != length(cone.idxs[k])
            error("dimension of cone $k does not match number of indices in the corresponding range")
        end
        qcone += dimension(cone.prmtvs[k])
    end
    if qcone != q
        error("dimension of cone is not consistent with number of rows in G and h")
    end

    return nothing
end

# preprocess data (optional)
function preprocess_data(
    P::AbstractMatrix{Float64},
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64};
    tol::Float64 = 1e-13, # presolve tolerance
    useQR::Bool = false, # returns QR fact of A' for use in a QR-based linear system solver
    )
    (n, p) = (length(c), length(b))
    q = size(G, 1)

    # NOTE (pivoted) QR factorizations are usually rank-revealing but may be unreliable, see http://www.math.sjsu.edu/~foster/rankrevealingcode.html
    # rank of a matrix is number of nonzero diagonal elements of R

    # preprocess dual equality constraints
    dukeep = 1:n
    PAG = vcat(P, A, G)

    # get pivoted QR # TODO when Julia has a unified QR interface, replace this
    if issparse(PAG)
        PAGF = qr(PAG, tol=tol)
    else
        PAGF = qr(PAG, Val(true))
    end
    PAGR = PAGF.R
    PAGrank = 0
    for i in 1:size(PAGR, 1) # TODO could replace this with rank(PAGF) when available for both dense and sparse
        if abs(PAGR[i,i]) > tol
            PAGrank += 1
        end
    end

    if PAGrank < n
        if issparse(PAG)
            dukeep = PAGF.pcol[1:PAGrank]
            PAGQ1 = Matrix{Float64}(undef, n+p+q, PAGrank)
            PAGQ1[PAGF.prow,:] = PAGF.Q*Matrix{Float64}(I, n+p+q, PAGrank) # TODO could eliminate this allocation
        else
            dukeep = PAGF.p[1:PAGrank]
            PAGQ1 = PAGF.Q*Matrix{Float64}(I, n+p+q, PAGrank) # TODO could eliminate this allocation
        end
        PAGRiQ1 = UpperTriangular(PAGR[1:PAGrank,1:PAGrank])\PAGQ1'

        P1 = P[dukeep,dukeep]
        A1 = A[:,dukeep]
        G1 = G[:,dukeep]
        c1 = c[dukeep]

        if norm(PAG'*PAGRiQ1'*c1 - c, Inf) > tol
            error("some dual equality constraints are inconsistent")
        end

        P = P1
        A = A1
        G = G1
        c = c1
        println("removed $(n - PAGrank) out of $n dual equality constraints")
        n = PAGrank
    end

    if p == 0
        # no primal equality constraints to preprocess
        # TODO use I instead of dense for Q2
        return (P, c, A, b, G, 1:0, dukeep, Matrix{Float64}(I, n, n), Matrix{Float64}(I, 0, n))
    end

    # preprocess primal equality constraints
    # get pivoted QR # TODO when Julia has a unified QR interface, replace this
    if issparse(A)
        AF = qr(sparse(A'), tol=tol)
    else
        AF = qr(A', Val(true))
    end
    AR = AF.R
    Arank = 0
    for i in 1:size(AR, 1) # TODO could replace this with rank(AF) when available for both dense and sparse
        if abs(AR[i,i]) > tol
            Arank += 1
        end
    end

    if !useQR && Arank == p
        # no primal equalities to remove and QR of A' not needed
        return (P, c, A, b, G, 1:p, dukeep, Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0))
    end

    # using QR of A' (requires reordering rows) and/or some primal equalities are dependent
    if issparse(A)
        prkeep = AF.pcol[1:Arank]
        AQ = Matrix{Float64}(undef, n, n)
        AQ[AF.prow,:] = AF.Q*Matrix{Float64}(I, n, n) # TODO could eliminate this allocation
    else
        prkeep = AF.p[1:Arank]
        AQ = AF.Q*Matrix{Float64}(I, n, n) # TODO could eliminate this allocation
    end
    AQ2 = AQ[:,Arank+1:n]
    ARiQ1 = UpperTriangular(AR[1:Arank,1:Arank])\AQ[:,1:Arank]'

    A1 = A[prkeep,:]
    b1 = b[prkeep]

    if Arank < p
        # some dependent primal equalities, so check if they are consistent
        x1 = ARiQ1'*b1
        if norm(A*x1 - b, Inf) > tol
            error("some primal equality constraints are inconsistent")
        end
        println("removed $(p - Arank) out of $p primal equality constraints")
    end

    return (P, c, A1, b1, G, prkeep, dukeep, AQ2, ARiQ1)
end

# verify problem data and load into model object
function load_data!(
    mdl::Model,
    P::AbstractMatrix{Float64},
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    cone::Cone,
    L::LinSysCache, # linear system solver cache (see linsyssolvers folder)
    )
    (mdl.P, mdl.c, mdl.A, mdl.b, mdl.G, mdl.h, mdl.cone, mdl.L) = (P, c, A, b, G, h, cone, L)
    mdl.status = :Loaded
    return mdl
end

# solve cone QP without homogeneous self-dual embedding
function solve!(mdl::Model)
    starttime = time()
    (P, c, A, b, G, h, cone, L) = (mdl.P, mdl.c, mdl.A, mdl.b, mdl.G, mdl.h, mdl.cone, mdl.L)
    (n, p, q) = (length(c), length(b), length(h))
    bnu = barrierpar(cone) # complexity parameter of the barrier (sum of the primitive cone barrier parameters)


    # values during line searches
    ls_z = similar(h)
    ls_s = similar(h)
    # cone functions evaluate barrier derivatives
    loadpnt!(cone, ls_s, ls_z)
    g = similar(h)


    # find initial primal-dual iterate
    mdl.verbose && println("\nfinding initial iterate")

    # TODO scale like in alfonso?
    getinitsz!(ls_s, ls_z, cone)
    s = copy(ls_s)
    z = copy(ls_z)

    mu = dot(z, s)/bnu
    @assert !isnan(mu)
    if abs(1.0 - mu) > 1e-6
        error("mu is $mu")
    end

    # solve for x and y
    # Px + A'y = -c - G'z
    # Ax = b
    # Gx = h - s
    # TODO do this more efficiently as a 3x3 system in the linsys solver files
    rhs = [-c - G'*z; b; h - s]
    if issparse(P) && issparse(A) && issparse(G)
        xy = [P A'; A spzeros(p, p); G spzeros(q, p)]\rhs
    else
        xy = [P A'; A zeros(p, p); G zeros(q, p)]\rhs
    end
    x = xy[1:n]
    y = xy[n+1:end]

    mdl.verbose && println("initial iterate found")


    # preallocate helper arrays
    tmp_x = similar(x)
    tmp_x2 = similar(x)
    tmp_y = similar(y)
    tmp_z = similar(z)
    tmp_s = similar(s)


    # calculate tolerances for convergence
    tol_res_x = inv(max(1.0, norm(c)))
    tol_res_y = inv(max(1.0, norm(b)))
    tol_res_z = inv(max(1.0, norm(h)))

    # calculate prediction and correction step parameters
    (beta, eta, cpredfix) = getbetaeta(mdl.maxcorrsteps, bnu) # beta: large neighborhood parameter, eta: small neighborhood parameter
    alphapredfix = cpredfix/(eta + sqrt(2.0*abs2(eta) + bnu)) # fixed predictor step size
    alphapredthres = (mdl.predlsmulti^mdl.maxpredsmallsteps)*alphapredfix # minimum predictor step size
    alphapredinit = (mdl.predlinesearch ? min(1e2*alphapredfix, 0.99999) : alphapredfix) # predictor step size

    # main loop
    if mdl.verbose
        println("starting iteration")
        @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s\n", "iter", "p_obj", "d_obj", "abs_gap", "rel_gap", "x_res", "y_res", "z_res", "mu")
        flush(stdout)
    end

    mdl.status = :StartedIterating
    alphapred = alphapredinit
    pobj = 0.0
    dobj = 0.0
    iter = 0
    while true
        # calculate residuals and convergence parameters
        # tmp_x = P*x + A'*y + G'*z + c
        tmp_x .= P*x + A'*y + G'*z + c
        # mul!(tmp_x2, A', y)
        # mul!(tmp_x, G', z)
        # @. tmp_x = -tmp_x2 - tmp_x - c
        nres_x = norm(tmp_x)

        # tmp_y = A*x - b
        mul!(tmp_y, A, x)
        @. tmp_y -= b
        nres_y = norm(tmp_y)

        # tmp_z = s + G*x - h
        mul!(tmp_z, G, x)
        @. tmp_z += s - h
        nres_z = norm(tmp_z)

        gap = dot(z, s) # TODO maybe should adapt original Alfonso condition instead of using this CVXOPT condition

        # TODO add objective constant
        pobj = 0.5*x'*P*x + dot(c, x) # TODO use Px calculated already for tmp_x
        dobj = pobj + dot(y, tmp_y) + dot(z, tmp_z) - gap
        # @assert dobj ≈ pobj + z'*(G*x - h) + y'*(A*x - b)

        # TODO maybe add small epsilon to denominators that are zero to avoid NaNs, and get rid of isnans further down
        if pobj < 0.0
            relgap = gap/-pobj
        elseif dobj > 0.0
            relgap = gap/dobj
        else
            relgap = NaN
        end

        xres = nres_x*tol_res_x
        yres = nres_y*tol_res_y
        zres = nres_z*tol_res_z

        if mdl.verbose
            # print iteration statistics
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n", iter, pobj, dobj, gap, relgap, xres, yres, zres, mu)
            flush(stdout)
        end

        # check convergence criteria
        # TODO nearly primal or dual infeasible or nearly optimal cases?
        if max(xres, yres, zres) <= mdl.tolfeas && (gap <= mdl.tolabsopt || (!isnan(relgap) && relgap <= mdl.tolrelopt))
            mdl.verbose && println("optimal solution found; terminating")
            mdl.status = :Optimal
            break
        # elseif !isnan(infres_pr) && infres_pr <= mdl.tolfeas
        #     mdl.verbose && println("primal infeasibility detected; terminating")
        #     mdl.status = :PrimalInfeasible
        #     break
        # elseif !isnan(infres_du) && infres_du <= mdl.tolfeas
        #     mdl.verbose && println("dual infeasibility detected; terminating")
        #     mdl.status = :DualInfeasible
        #     break
        end

        # check iteration limit
        iter += 1
        if iter >= mdl.maxiter
            mdl.verbose && println("iteration limit reached; terminating")
            mdl.status = :IterationLimit
            break
        end

        # check time limit
        if (time() - starttime) >= mdl.timelimit
            mdl.verbose && println("time limit reached; terminating")
            mdl.status = :TimeLimit
            break
        end

        # prediction phase
        # calculate prediction direction

        # TODO reverse sign of residuals above
        tmp_x .*= -1.0
        tmp_y .*= -1.0
        tmp_z .*= -1.0
        for k in eachindex(cone.prmtvs)
            vk = view(cone.prmtvs[k].usedual ? s : z, cone.idxs[k])
            tmp_s[cone.idxs[k]] = -vk
        end
        solvelinsys4!(tmp_x, tmp_y, tmp_z, tmp_s, mu, L)

        # determine step length alpha by line search
        alpha = alphapred

        nbhd = Inf
        ls_mu = 0.0
        alphaprevok = true
        predfail = false
        nprediters = 0
        while true
            nprediters += 1

            @. ls_z = z + alpha*tmp_z
            @. ls_s = s + alpha*tmp_s
            ls_mu = dot(ls_s, ls_z)/bnu

            # accept primal iterate if
            # - decreased alpha and it is the first inside the cone and beta-neighborhood or
            # - increased alpha and it is inside the cone and the first to leave beta-neighborhood
            if ls_mu > 0.0 && incone(cone, mu)
                # primal iterate is inside the cone
                nbhd = calcnbhd!(g, ls_s, ls_z, ls_mu, cone)
                # @show sqrt(nbhd)/ls_mu, beta
                if nbhd < abs2(beta*ls_mu)
                    # iterate is inside the beta-neighborhood
                    if !alphaprevok || alpha > mdl.predlsmulti
                        # either the previous iterate was outside the beta-neighborhood or increasing alpha again will make it > 1
                        if mdl.predlinesearch
                            alphapred = alpha
                        end
                        break
                    end

                    alphaprevok = true
                    alpha = alpha/mdl.predlsmulti # increase alpha
                    continue
                end
            end

            # primal iterate is either
            # - outside the cone or
            # - inside the cone and outside the beta-neighborhood and previous iterate was outside the beta-neighborhood
            if alpha < alphapredthres
                # alpha is very small, so predictor has failed
                predfail = true
                mdl.verbose && println("predictor could not improve the solution ($nprediters line search steps); terminating")
                break
            end

            alphaprevok = false
            alpha = mdl.predlsmulti*alpha # decrease alpha
        end
        if predfail
            mdl.status = :PredictorFail
            break
        end

        # step distance alpha in the direction
        @. x += alpha*tmp_x
        @. y += alpha*tmp_y
        @. ls_z = z + alpha*tmp_z
        @. ls_s = s + alpha*tmp_s
        @. z = ls_z
        @. s = ls_s
        mu = ls_mu

        # @show mu
        # @show sqrt(nbhd)/mu

        # skip correction phase if allowed and current iterate is in the eta-neighborhood
        if mdl.corrcheck && nbhd <= abs2(eta*mu)
            continue
        end

        # correction phase
        corrfail = false
        ncorrsteps = 0
        while true
            ncorrsteps += 1

            # calculate correction direction
            calcg!(g, cone)
            tmp_x .= 0.0
            tmp_y .= 0.0
            tmp_z .= 0.0
            for k in eachindex(cone.prmtvs)
                vk = view(cone.prmtvs[k].usedual ? s : z, cone.idxs[k])
                gk = view(g, cone.idxs[k])
                @. tmp_s[cone.idxs[k]] = -vk - mu * gk
            end
            solvelinsys4!(tmp_x, tmp_y, tmp_z, tmp_s, mu, L)

            # determine step length alpha by line search
            alpha = mdl.alphacorr
            ncorrlsiters = 0
            while ncorrlsiters <= mdl.maxcorrlsiters
                ncorrlsiters += 1

                @. ls_z = z + alpha*tmp_z
                @. ls_s = s + alpha*tmp_s
                ls_mu = dot(ls_s, ls_z)/bnu

                if ls_mu > 0.0 && incone(cone, mu)
                    # primal iterate x is inside the cone, so terminate line search
                    break
                end

                # primal iterate x is outside the cone
                if ncorrlsiters == mdl.maxcorrlsiters
                    # corrector failed
                    corrfail = true
                    mdl.verbose && println("corrector could not improve the solution ($ncorrlsiters line search steps); terminating")
                    break
                end

                alpha = mdl.corrlsmulti*alpha # decrease alpha
            end
            if corrfail
                break
            end

            # @show alpha

            # step distance alpha in the direction
            @. x += alpha*tmp_x
            @. y += alpha*tmp_y
            @. z = ls_z
            @. s = ls_s
            mu = ls_mu

            # finish if allowed and current iterate is in the eta-neighborhood, or if taken max steps
            if ncorrsteps == mdl.maxcorrsteps || mdl.corrcheck
                nbhd = calcnbhd!(g, ls_s, ls_z, mu, cone)
                @. ls_z = z
                @. ls_s = s

                # @show mu
                # @show sqrt(nbhd)/mu

                if nbhd <= abs2(eta*mu)
                    break
                elseif ncorrsteps == mdl.maxcorrsteps
                    # outside eta neighborhood, so corrector failed
                    corrfail = true
                    mdl.verbose && println("corrector phase finished outside the eta-neighborhood ($ncorrsteps correction steps); terminating")
                    break
                end
            end
        end
        if corrfail
            mdl.status = :CorrectorFail
            break
        end
    end

    # store result and iteration statistics
    mdl.pobj = pobj
    mdl.dobj = dobj
    mdl.x = x
    mdl.s = s
    mdl.y = y
    mdl.z = z
    mdl.mu = mu
    mdl.niters = iter
    mdl.solvetime = time() - starttime
    mdl.verbose && println("\nstatus is $(mdl.status) after $(mdl.niters) iterations and $(trunc(mdl.solvetime, digits=3)) seconds\n")

    return nothing
end

# get neighborhood parameters depending on magnitude of barrier parameter and maximum number of correction steps
# TODO calculate values from the formulae given in Papp & Yildiz "On A Homogeneous Interior-Point Algorithm for Non-Symmetric Convex Conic Optimization"
function getbetaeta(maxcorrsteps::Int, bnu::Float64)
    if maxcorrsteps <= 2
        if bnu < 10.0
            return (0.1810, 0.0733, 0.0225)
        elseif bnu < 100.0
            return (0.2054, 0.0806, 0.0263)
        else
            return (0.2190, 0.0836, 0.0288)
        end
    elseif maxcorrsteps <= 4
        if bnu < 10.0
            return (0.2084, 0.0502, 0.0328)
        elseif bnu < 100.0
            return (0.2356, 0.0544, 0.0380)
        else
            return (0.2506, 0.0558, 0.0411)
        end
    else
        if bnu < 10.0
            return (0.2387, 0.0305, 0.0429)
        elseif bnu < 100.0
            return (0.2683, 0.0327, 0.0489)
        else
            return (0.2844, 0.0332, 0.0525)
        end
    end
end




# TODO delete later


# # TODO use method in cvxopt ch 5.3
# if issparse(P) && issparse(A) && issparse(G)
#     LHS = [P A' G'; A spzeros(p, p+q); G spzeros(q, p) -1.0I]
#     @assert issparse(LHS)
# else
#     LHS = [Matrix(P) A' G'; A zeros(p, p+q); G zeros(q, p) -1.0I]
#     @assert !issparse(LHS)
# end
# rhs = [-c; b; h]
# soln = Symmetric(LHS)\rhs
#
# x = soln[1:n]
# y = soln[n+1:n+p]
# z = soln[n+p+1:end]
#
# ls_s = -z
# ls_z = copy(z)
# loadpnt!(cone, ls_s, ls_z)
#
# tmp_s = similar(z)
# tmp_z = similar(z)
# for k in eachindex(cone.prmtvs)
#     v1k = view((cone.prmtvs[k].usedual ? ls_s : ls_z), cone.idxs[k])
#     v2k = view((cone.prmtvs[k].usedual ? ls_z : ls_s), cone.idxs[k])
#
#     if !incone_prmtv(cone.prmtvs[k], 1.0)
#         getintdir_prmtv!(v1k, cone.prmtvs[k])
#         @. v2k += v1k
#         steps = 1
#         alpha = 1.0
#         while !incone_prmtv(cone.prmtvs[k], 1.0)
#             @. v2k += alpha * v1k
#             steps += 1
#             if steps > 25
#                 error("cannot find initial iterate")
#             end
#             alpha *= 1.5
#         end
#         @show k, steps, alpha
#     end
#
#     calcg_prmtv!(v1k, cone.prmtvs[k])
#     @. v1k = -v1k
# end
#
# @assert incone(cone, 1.0) # TODO delete
#
# s = copy(ls_s)
# @. z = ls_z
#
# mu = dot(z, s)/bnu
# @assert !isnan(mu)
# if abs(1.0 - mu) > 1e-6
#     error("mu is $mu")
# end
#
# g = similar(z)
# @assert calcnbhd!(g, copy(s), copy(z), mu, cone) < 1e-6



# @. ls_z = z
# @. ls_s = s

# @. tmp_s = tmp_z
# for k in eachindex(cone.prmtvs)
#     v1 = (cone.prmtvs[k].usedual ? s : z)
#     @. @views tmp_z[cone.idxs[k]] = -v1[cone.idxs[k]]
# end

# @. @views begin
#     rhs[1:n] = -tmp_x
#     rhs[n+1:n+p] = -tmp_y
#     rhs[n+p+1:end] = -tmp_z + s
# end
#
# for k in eachindex(cone.prmtvs)
#     idxs = (n + p) .+ cone.idxs[k]
#     dim = dimension(cone.prmtvs[k])
#     Hk = view(LHS, idxs, idxs)
#
#     if cone.prmtvs[k].usedual
#         @. Hk = -mu * cone.prmtvs[k].H # NOTE only upper triangle is valid
#         # calcHarr_prmtv!(Hk, Matrix(-mu*I, dim, dim), cone.prmtvs[k])
#     else
#         Hinv = inv(cone.prmtvs[k].F)
#         @. Hk = Hinv / (-mu)
#         # calcHiarr_prmtv!(Hk, Matrix(-inv(mu)*I, dim, dim), cone.prmtvs[k])
#     end
# end
#
# soln = Symmetric(LHS, :U)\rhs
# @. @views begin
#     tmp_x = soln[1:n]
#     tmp_y = soln[n+1:n+p]
#     tmp_s = -tmp_z + s
#     tmp_z = soln[n+p+1:end]
# end
# tmp_s -= G*tmp_x



# solvelinsys6!(tmp_x, tmp_y, tmp_z, -kap, tmp_s, kap + cx + by + hz, mu, tau, L)

# # check residual
# res_x = -A'*tmp_y - G'*tmp_z - c*tmp_au + copy_x
# res_y = A*tmp_x - b*tmp_au + copy_y
# # res_z = tmp_s + G*tmp_x - h*tmp_au - copy_z
# res_obj = dot(c, tmp_x) + dot(b, tmp_y) + dot(h, tmp_z) + tmp_kap + (kap + cx + by + hz)
#
# @show norm(res_x)
# @show norm(res_y)
# @show norm(res_z)
# @show norm(res_obj)



# calcg!(g, cone)
#
# for k in eachindex(cone.prmtvs)
#     idxs = (n + p + q) .+ cone.idxs[k]
#     LHS4[idxs, idxs] = mu * Symmetric(cone.prmtvs[k].H)
#
#     # Hk = view(LHS4, (n + p) .+ cone.idxs[k], (n + p + q) .+ cone.idxs[k])
#     # dim = dimension(cone.prmtvs[k])
#     # calcHarr_prmtv!(Hk, Matrix(mu*I, dim, dim), cone.prmtvs[k])
# end
#
# rhs4 = [zeros(n); zeros(p); zeros(q); -z - mu*g]

# LHS4 = [
#     P           A'          G'                zeros(n,q)       ;
#     A           zeros(p,p)  zeros(p,q)        zeros(p,q)       ;
#     G           zeros(q,p)  zeros(q,q)        Matrix(1.0I,q,q) ;
#     zeros(q,n)  zeros(q,p)  Matrix(1.0I,q,q)  Matrix(1.0I,q,q) ;
#     ]




#
# calcg!(g, cone)
# @. @views begin
#     rhs[1:n] = 0.0
#     rhs[n+1:n+p] = 0.0
#     rhs[n+p+1:end] = z + mu*g
# end
#
# for k in eachindex(cone.prmtvs)
#     idxs = (n + p) .+ cone.idxs[k]
#     # dim = dimension(cone.prmtvs[k])
#     # Hk = view(LHS, idxs, idxs)
#
#     if cone.prmtvs[k].usedual
#         @. LHS[idxs, idxs] = -mu * cone.prmtvs[k].H # NOTE only upper triangle is valid
#         # calcHarr_prmtv!(Hk, Matrix(-mu*I, dim, dim), cone.prmtvs[k])
#     else
#         Hinv = inv(cone.prmtvs[k].F)
#         @. LHS[idxs, idxs] = Hinv / (-mu)
#         # calcHiarr_prmtv!(Hk, Matrix(-inv(mu)*I, dim, dim), cone.prmtvs[k])
#     end
# end
#
# soln = Symmetric(LHS, :U)\rhs
# @. @views begin
#     tmp_x = soln[1:n]
#     tmp_y = soln[n+1:n+p]
#     tmp_s = -z - mu*g
#     tmp_z = soln[n+p+1:end]
# end
# tmp_s -= G*tmp_x
#
#



# (tmp_kap, tmp_tau) = solvelinsys6!(tmp_x, tmp_y, tmp_z, -kap + mu/tau, tmp_s, 0.0, mu, tau, L)
