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

    x::Vector{Float64}      # final value of the primal free variables
    s::Vector{Float64}      # final value of the primal cone variables
    y::Vector{Float64}      # final value of the dual free variables
    z::Vector{Float64}      # final value of the dual cone variables
    tau::Float64            # final value of the tau variable
    kap::Float64            # final value of the kappa variable
    mu::Float64             # final value of mu
    pobj::Float64           # final primal objective value
    dobj::Float64           # final dual objective value

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

get_x(mdl::Model) = copy(mdl.x)
get_s(mdl::Model) = copy(mdl.s)
get_y(mdl::Model) = copy(mdl.y)
get_z(mdl::Model) = copy(mdl.z)

get_tau(mdl::Model) = mdl.tau
get_kappa(mdl::Model) = mdl.kappa
get_mu(mdl::Model) = mdl.mu

get_pobj(mdl::Model) = dot(mdl.c, mdl.x)
get_dobj(mdl::Model) = -dot(mdl.b, mdl.y) - dot(mdl.h, mdl.z)

# check data for consistency
function check_data(
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    cone::Cone,
    )
    (n, p, q) = (length(c), length(b), length(h))
    if n == 0
        error("c vector is empty, but number of variables must be positive")
    end
    if q == 0
        println("no conic constraints were specified; proceeding anyway")
    end
    if n < p
        println("number of equality constraints ($p) exceeds number of variables ($n)")
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
    AG = vcat(A, G)

    # get pivoted QR # TODO when Julia has a unified QR interface, replace this
    if issparse(AG)
        AGF = qr(AG, tol=tol)
    else
        AGF = qr(AG, Val(true))
    end
    AGR = AGF.R
    AGrank = 0
    for i in 1:size(AGR, 1) # TODO could replace this with rank(AF) when available for both dense and sparse
        if abs(AGR[i,i]) > tol
            AGrank += 1
        end
    end

    if AGrank < n
        if issparse(AG)
            dukeep = AGF.pcol[1:AGrank]
            AGQ1 = Matrix{Float64}(undef, p + q, AGrank)
            AGQ1[AGF.prow,:] = AGF.Q*Matrix{Float64}(I, p + q, AGrank) # TODO could eliminate this allocation
        else
            dukeep = AGF.p[1:AGrank]
            AGQ1 = AGF.Q*Matrix{Float64}(I, p + q, AGrank) # TODO could eliminate this allocation
        end
        AGRiQ1 = UpperTriangular(AGR[1:AGrank,1:AGrank])\AGQ1'

        A1 = A[:,dukeep]
        G1 = G[:,dukeep]
        c1 = c[dukeep]

        if norm(AG'*AGRiQ1'*c1 - c, Inf) > tol
            error("some dual equality constraints are inconsistent")
        end

        A = A1
        G = G1
        c = c1
        println("removed $(n - AGrank) out of $n dual equality constraints")
        n = AGrank
    end

    if p == 0
        # no primal equality constraints to preprocess
        # TODO use I instead of dense for Q2
        return (c, A, b, G, 1:0, dukeep, Matrix{Float64}(I, n, n), Matrix{Float64}(I, 0, n))
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
        return (c, A, b, G, 1:p, dukeep, Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0))
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

    A = A1
    b = b1
    return (c, A, b, G, prkeep, dukeep, AQ2, ARiQ1)
end

# verify problem data and load into model object
function load_data!(
    mdl::Model,
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    cone::Cone,
    L::LinSysCache, # linear system solver cache (see linsyssolvers folder)
    )
    (mdl.c, mdl.A, mdl.b, mdl.G, mdl.h, mdl.cone, mdl.L) = (c, A, b, G, h, cone, L)
    mdl.status = :Loaded
    return mdl
end

# solve using predictor-corrector algorithm based on homogeneous self-dual embedding
function solve!(mdl::Model)
    starttime = time()
    (c, A, b, G, h, cone, L) = (mdl.c, mdl.A, mdl.b, mdl.G, mdl.h, mdl.cone, mdl.L)
    (n, p, q) = (length(c), length(b), length(h))
    bnu = 1.0 + barrierpar(cone) # complexity parameter nu-bar of the augmented barrier (sum of the primitive cone barrier parameters plus 1)

    # preallocate arrays
    # primal and dual variables multiplied by tau
    tx = similar(c)
    ty = similar(b)
    tz = similar(h)
    ts = similar(h)
    # values during line searches
    ls_tz = similar(tz)
    ls_ts = similar(ts)
    # cone functions evaluate barrier derivatives
    loadpnt!(cone, ls_ts, ls_tz)
    g = similar(ts)
    # helper arrays for residuals, right-hand-sides, and search directions
    tmp_tx = similar(tx)
    tmp_tx2 = similar(tx)
    tmp_ty = similar(ty)
    tmp_tz = similar(tz)
    tmp_ts = similar(ts)

    # find initial primal-dual iterate
    mdl.verbose && println("\nfinding initial iterate")

    # TODO scale like in alfonso?
    getinitsz!(ls_ts, ls_tz, cone)
    @. ts = ls_ts
    @. tz = ls_tz

    tau = 1.0
    kap = 1.0
    mu = (dot(tz, ts) + tau*kap)/bnu
    @assert !isnan(mu)
    if abs(1.0 - mu) > 1e-6
        error("mu is $mu")
    end

    # solve for tx and ty
    # A'y = -c - G'z
    # Ax = b
    # Gx = h - ts
    # TODO do this more efficiently as a 3x3 system in the linsys solver files
    rhs = [-c - G'*tz; b; h - ts]
    if issparse(A) && issparse(G)
        txty = [spzeros(n, n) A'; A spzeros(p, p); G spzeros(q, p)]\rhs
    else
        txty = [zeros(n, n) A'; A zeros(p, p); G zeros(q, p)]\rhs
    end
    @. @views tx = txty[1:n]
    @. @views ty = txty[n+1:end]

    rhs_tx = -c - G'*tz
    rhs_ty = b
    rhs_tz = h - ts
    solvelinsys3!(rhs_tx, rhs_ty, rhs_tz, L)

    @show


    mdl.verbose && println("initial iterate found")

    # calculate tolerances for convergence
    tol_res_tx = inv(max(1.0, norm(c)))
    tol_res_ty = inv(max(1.0, norm(b)))
    tol_res_tz = inv(max(1.0, norm(h)))

    # calculate prediction and correction step parameters
    (beta, eta, cpredfix) = getbetaeta(mdl.maxcorrsteps, bnu) # beta: large neighborhood parameter, eta: small neighborhood parameter
    alphapredfix = cpredfix/(eta + sqrt(2.0*abs2(eta) + bnu)) # fixed predictor step size
    alphapredthres = (mdl.predlsmulti^mdl.maxpredsmallsteps)*alphapredfix # minimum predictor step size
    alphapredinit = (mdl.predlinesearch ? min(1e2*alphapredfix, 0.99999) : alphapredfix) # predictor step size

    # main loop
    if mdl.verbose
        println("starting iteration")
        @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s\n", "iter", "p_obj", "d_obj", "abs_gap", "rel_gap", "p_inf", "d_inf", "tau", "kap", "mu")
        flush(stdout)
    end

    mdl.status = :StartedIterating
    alphapred = alphapredinit
    iter = 0
    while true
        # calculate residuals and convergence parameters
        # tmp_tx = -A'*ty - G'*tz - c*tau
        mul!(tmp_tx2, A', ty)
        mul!(tmp_tx, G', tz)
        @. tmp_tx = -tmp_tx2 - tmp_tx
        nres_x = norm(tmp_tx)
        @. tmp_tx -= c*tau
        nres_tx = norm(tmp_tx)/tau

        # tmp_ty = A*tx - b*tau
        mul!(tmp_ty, A, tx)
        nres_y = norm(tmp_ty)
        @. tmp_ty -= b*tau
        nres_ty = norm(tmp_ty)/tau

        # tmp_tz = ts + G*tx - h*tau
        mul!(tmp_tz, G, tx)
        @. tmp_tz += ts
        nres_z = norm(tmp_tz)
        @. tmp_tz -= h*tau
        nres_tz = norm(tmp_tz)/tau

        (cx, by, hz) = (dot(c, tx), dot(b, ty), dot(h, tz))
        obj_pr = cx/tau
        obj_du = -(by + hz)/tau
        gap = dot(tz, ts) # TODO maybe should adapt original Alfonso condition instead of using this CVXOPT condition

        # TODO maybe add small epsilon to denominators that are zero to avoid NaNs, and get rid of isnans further down
        if obj_pr < 0.0
            relgap = gap/-obj_pr
        elseif obj_du > 0.0
            relgap = gap/obj_du
        else
            relgap = NaN
        end

        nres_pr = max(nres_ty*tol_res_ty, nres_tz*tol_res_tz)
        nres_du = nres_tx*tol_res_tx

        if hz + by < 0.0
            infres_pr = nres_x*tol_res_tx/(-hz - by)
        else
            infres_pr = NaN
        end
        if cx < 0.0
            infres_du = -max(nres_y*tol_res_ty, nres_z*tol_res_tz)/cx
        else
            infres_du = NaN
        end

        if mdl.verbose
            # print iteration statistics
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n", iter, obj_pr, obj_du, gap, relgap, nres_pr, nres_du, tau, kap, mu)
            flush(stdout)
        end

        # check convergence criteria
        # TODO nearly primal or dual infeasible or nearly optimal cases?
        if nres_pr <= mdl.tolfeas && nres_du <= mdl.tolfeas && (gap <= mdl.tolabsopt || (!isnan(relgap) && relgap <= mdl.tolrelopt))
            mdl.verbose && println("optimal solution found; terminating")
            mdl.status = :Optimal
            break
        elseif !isnan(infres_pr) && infres_pr <= mdl.tolfeas
            mdl.verbose && println("primal infeasibility detected; terminating")
            mdl.status = :PrimalInfeasible
            break
        elseif !isnan(infres_du) && infres_du <= mdl.tolfeas
            mdl.verbose && println("dual infeasibility detected; terminating")
            mdl.status = :DualInfeasible
            break
        elseif mu <= mdl.tolfeas*1e-2 && tau <= mdl.tolfeas*1e-2*min(1.0, kap)
            mdl.verbose && println("ill-posedness detected; terminating")
            mdl.status = :IllPosed
            break
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
        @. ls_tz = tz
        @. ls_ts = ts
        @. tmp_ts = tmp_tz
        for k in eachindex(cone.prmtvs)
            v1 = (cone.prmtvs[k].usedual ? ts : tz)
            @. @views tmp_tz[cone.idxs[k]] = -v1[cone.idxs[k]]
        end

        # copy_x = copy(tmp_tx)
        # copy_y = copy(tmp_ty)
        # copy_z = copy(tmp_tz)

        (tmp_kap, tmp_tau) = solvelinsys6!(tmp_tx, tmp_ty, tmp_tz, -kap, tmp_ts, kap + cx + by + hz, mu, tau, L)

        # # check residual
        # res_x = -A'*tmp_ty - G'*tmp_tz - c*tmp_tau + copy_x
        # res_y = A*tmp_tx - b*tmp_tau + copy_y
        # # res_z = tmp_ts + G*tmp_tx - h*tmp_tau - copy_z
        # res_obj = dot(c, tmp_tx) + dot(b, tmp_ty) + dot(h, tmp_tz) + tmp_kap + (kap + cx + by + hz)
        #
        # @show norm(res_x)
        # @show norm(res_y)
        # # @show norm(res_z)
        # @show norm(res_obj)

        # determine step length alpha by line search
        alpha = alphapred
        if tmp_kap < 0.0
            alpha = min(alpha, -kap/tmp_kap*0.99999)
        end
        if tmp_tau < 0.0
            alpha = min(alpha, -tau/tmp_tau*0.99999)
        end

        nbhd = Inf
        ls_tau = ls_kap = ls_tk = ls_mu = 0.0
        alphaprevok = true
        predfail = false
        nprediters = 0
        while true
            nprediters += 1

            @. ls_tz = tz + alpha*tmp_tz
            @. ls_ts = ts + alpha*tmp_ts
            ls_tau = tau + alpha*tmp_tau
            ls_kap = kap + alpha*tmp_kap
            ls_tk = ls_tau*ls_kap
            ls_mu = (dot(ls_ts, ls_tz) + ls_tk)/bnu

            # accept primal iterate if
            # - decreased alpha and it is the first inside the cone and beta-neighborhood or
            # - increased alpha and it is inside the cone and the first to leave beta-neighborhood
            if ls_mu > 0.0 && ls_tau > 0.0 && ls_kap > 0.0 && incone(cone, ls_mu)
                # primal iterate is inside the cone
                nbhd = calcnbhd!(g, ls_ts, ls_tz, ls_mu, cone) + abs2(ls_tk - ls_mu)
                # nbhd = calcnbhd!(g, ls_ts, ls_tz, ls_mu, cone) + abs2(ls_tk - ls_mu)/abs2(ls_mu)

                if nbhd < abs2(beta*ls_mu)
                # if nbhd < abs2(beta)
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
        @. tx += alpha*tmp_tx
        @. ty += alpha*tmp_ty
        @. ls_tz = tz + alpha*tmp_tz
        @. ls_ts = ts + alpha*tmp_ts
        @. tz = ls_tz
        @. ts = ls_ts
        tau = ls_tau
        kap = ls_kap
        mu = ls_mu

        # skip correction phase if allowed and current iterate is in the eta-neighborhood
        if mdl.corrcheck && nbhd <= abs2(eta*mu)
        # if mdl.corrcheck && nbhd <= abs2(eta)
            continue
        end

        # correction phase
        corrfail = false
        ncorrsteps = 0
        while true
            ncorrsteps += 1

            # calculate correction direction
            @. tmp_tx = 0.0
            @. tmp_ty = 0.0
            for k in eachindex(cone.prmtvs)
                v1 = (cone.prmtvs[k].usedual ? ts : tz)
                @. @views tmp_tz[cone.idxs[k]] = -v1[cone.idxs[k]]
            end
            calcg!(g, cone)
            @. tmp_tz -= mu*g
            # @. tmp_tz -= g
            @. tmp_ts = 0.0

            (tmp_kap, tmp_tau) = solvelinsys6!(tmp_tx, tmp_ty, tmp_tz, -kap + mu/tau, tmp_ts, 0.0, mu, tau, L)

            # determine step length alpha by line search
            alpha = mdl.alphacorr
            if tmp_kap < 0.0
                alpha = min(alpha, -kap/tmp_kap*0.99999)
            end
            if tmp_tau < 0.0
                alpha = min(alpha, -tau/tmp_tau*0.99999)
            end

            ncorrlsiters = 0
            while ncorrlsiters <= mdl.maxcorrlsiters
                ncorrlsiters += 1

                @. ls_tz = tz + alpha*tmp_tz
                @. ls_ts = ts + alpha*tmp_ts
                ls_tau = tau + alpha*tmp_tau
                @assert ls_tau > 0.0
                ls_kap = kap + alpha*tmp_kap
                @assert ls_kap > 0.0
                ls_mu = (dot(ls_ts, ls_tz) + ls_tau*ls_kap)/bnu

                if ls_mu > 0.0 && incone(cone, ls_mu)
                    # primal iterate tx is inside the cone, so terminate line search
                    break
                end

                # primal iterate tx is outside the cone
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

            # step distance alpha in the direction
            @. tx += alpha*tmp_tx
            @. ty += alpha*tmp_ty
            @. tz = ls_tz
            @. ts = ls_ts
            tau = ls_tau
            kap = ls_kap
            mu = ls_mu

            # finish if allowed and current iterate is in the eta-neighborhood, or if taken max steps
            if ncorrsteps == mdl.maxcorrsteps || mdl.corrcheck
                nbhd = calcnbhd!(g, ls_ts, ls_tz, mu, cone) + abs2(tau*kap - mu)
                # nbhd = calcnbhd!(g, ls_ts, ls_tz, mu, cone) + abs2(tau*kap - mu)/abs2(mu)

                @. ls_tz = tz
                @. ls_ts = ts

                if nbhd <= abs2(eta*mu)
                # if nbhd <= abs2(eta)
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

    # calculate result and iteration statistics
    mdl.x = tx ./= tau
    mdl.s = ts ./= tau
    mdl.y = ty ./= tau
    mdl.z = tz ./= tau
    mdl.tau = tau
    mdl.kap = kap
    mdl.mu = mu
    mdl.niters = iter
    mdl.solvetime = time() - starttime

    mdl.verbose && println("\nstatus is $(mdl.status) after $iter iterations and $(trunc(mdl.solvetime, digits=3)) seconds\n")

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
