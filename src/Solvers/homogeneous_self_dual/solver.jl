#=
Copyright 2018, Chris Coey and contributors

interior point type and functions for algorithms based on homogeneous self dual embedding

TODO make internal statuses types
=#

mutable struct HSDSolver <: Solver
    model::Models.LinearModel

    stepper::HSDStepper

    verbose::Bool
    tol_rel_opt::Float64
    tol_abs_opt::Float64
    tol_feas::Float64
    tol_slow::Float64
    max_iters::Int
    time_limit::Float64

    x_conv_tol::Float64
    y_conv_tol::Float64
    z_conv_tol::Float64

    point::Models.Point
    tau::Float64
    kap::Float64
    mu::Float64

    x_residual::Vector{Float64}
    y_residual::Vector{Float64}
    z_residual::Vector{Float64}
    x_norm_res_t::Float64
    y_norm_res_t::Float64
    z_norm_res_t::Float64
    x_norm_res::Float64
    y_norm_res::Float64
    z_norm_res::Float64

    primal_obj_t::Float64
    dual_obj_t::Float64
    primal_obj::Float64
    dual_obj::Float64
    gap::Float64
    rel_gap::Float64
    x_feas::Float64
    y_feas::Float64
    z_feas::Float64

    slow_prev::Bool
    prev_gap::Float64
    prev_rel_gap::Float64
    prev_x_feas::Float64
    prev_y_feas::Float64
    prev_z_feas::Float64

    status::Symbol
    num_iters::Int
    solve_time::Float64

    function HSDSolver(
        model::Models.LinearModel,
        ;
        stepper::HSDStepper = CombinedHSDStepper(model),
        verbose::Bool = true,
        tol_rel_opt = 1e-6,
        tol_abs_opt = 1e-7,
        tol_feas = 1e-7,
        tol_slow = 2e-2,
        max_iters::Int = 500,
        time_limit::Float64 = 3e2,
        )
        solver = new()

        solver.model = model

        solver.stepper = stepper

        solver.verbose = verbose
        solver.tol_rel_opt = tol_rel_opt
        solver.tol_abs_opt = tol_abs_opt
        solver.tol_feas = tol_feas
        solver.tol_slow = tol_slow
        solver.max_iters = max_iters
        solver.time_limit = time_limit

        solver.x_conv_tol = inv(max(1.0, norm(model.c)))
        solver.y_conv_tol = inv(max(1.0, norm(model.b)))
        solver.z_conv_tol = inv(max(1.0, norm(model.h)))

        solver.point = model.initial_point
        solver.tau = 1.0
        solver.kap = 1.0
        solver.mu = NaN

        solver.x_residual = similar(model.c)
        solver.y_residual = similar(model.b)
        solver.z_residual = similar(model.h)
        solver.x_norm_res_t = NaN
        solver.y_norm_res_t = NaN
        solver.z_norm_res_t = NaN
        solver.x_norm_res = NaN
        solver.y_norm_res = NaN
        solver.z_norm_res = NaN

        solver.primal_obj_t = NaN
        solver.dual_obj_t = NaN
        solver.primal_obj = NaN
        solver.dual_obj = NaN
        solver.gap = NaN
        solver.rel_gap = NaN
        solver.x_feas = NaN
        solver.y_feas = NaN
        solver.z_feas = NaN

        solver.slow_prev = false
        solver.prev_gap = NaN
        solver.prev_rel_gap = NaN
        solver.prev_x_feas = NaN
        solver.prev_y_feas = NaN
        solver.prev_z_feas = NaN

        solver.status = :SolveNotCalled
        solver.num_iters = 0
        solver.solve_time = NaN
        solver.primal_obj = NaN
        solver.dual_obj = NaN

        return solver
    end
end

get_tau(solver::HSDSolver) = solver.tau
get_kappa(solver::HSDSolver) = solver.kap
get_mu(solver::HSDSolver) = solver.mu

# TODO maybe use iteration interface rather than while loop
function solve(solver::HSDSolver)
    solver.status = :SolveCalled
    start_time = time()

    calc_mu(solver)
    if isnan(solver.mu) || abs(1.0 - solver.mu) > 1e-6
        error("initial mu is $(solver.mu) (should be 1.0)")
    end

    while true
        # @timeit "res" begin
        calc_residual(solver)
        # end

        # @timeit "conv" begin
        calc_convergence_params(solver)
        # end

        solver.verbose && print_iteration_stats(solver, solver.stepper)

        # @timeit "check" begin
        check_convergence(solver) && break
        # end

        if solver.num_iters == solver.max_iters
            solver.verbose && println("iteration limit reached; terminating")
            solver.status = :IterationLimit
            break
        end

        if time() - start_time >= solver.time_limit
            solver.verbose && println("time limit reached; terminating")
            solver.status = :TimeLimit
            break
        end

        # TODO may use different function, or function could change during some iteration eg if numerical difficulties
        # @timeit "step" begin
        step(solver, solver.stepper)
        # end

        solver.num_iters += 1
    end

    # calculate result and iteration statistics and finish
    point = solver.point
    point.x ./= solver.tau
    point.y ./= solver.tau
    point.z ./= solver.tau
    point.s ./= solver.tau

    solver.solve_time = time() - start_time

    solver.verbose && println("\nstatus is $(solver.status) after $(solver.num_iters) iterations and $(trunc(solver.solve_time, digits=3)) seconds\n")

    return
end

function calc_convergence_params(solver::HSDSolver)
    model = solver.model
    point = solver.point

    solver.prev_gap = solver.gap
    solver.prev_rel_gap = solver.rel_gap
    solver.prev_x_feas = solver.x_feas
    solver.prev_y_feas = solver.y_feas
    solver.prev_z_feas = solver.z_feas

    solver.primal_obj_t = dot(model.c, point.x)
    solver.dual_obj_t = -dot(model.b, point.y) - dot(model.h, point.z)
    solver.primal_obj = solver.primal_obj_t / solver.tau
    solver.dual_obj = solver.dual_obj_t / solver.tau
    solver.gap = dot(point.z, point.s) # TODO maybe should adapt original Alfonso condition instead of using this CVXOPT condition
    if solver.primal_obj < 0.0
        solver.rel_gap = solver.gap / -solver.primal_obj
    elseif solver.dual_obj > 0.0
        solver.rel_gap = solver.gap / solver.dual_obj
    else
        solver.rel_gap = NaN
    end
    solver.x_feas = solver.x_norm_res * solver.x_conv_tol
    solver.y_feas = solver.y_norm_res * solver.y_conv_tol
    solver.z_feas = solver.z_norm_res * solver.z_conv_tol

    return
end

function check_convergence(solver::HSDSolver)
    # check convergence criteria
    # TODO nearly primal or dual infeasible or nearly optimal cases?
    if max(solver.x_feas, solver.y_feas, solver.z_feas) <= solver.tol_feas &&
        (solver.gap <= solver.tol_abs_opt || (!isnan(solver.rel_gap) && solver.rel_gap <= solver.tol_rel_opt))
        solver.verbose && println("optimal solution found; terminating")
        solver.status = :Optimal
        return true
    end
    if solver.dual_obj_t > 0.0
        infres_pr = solver.x_norm_res_t * solver.x_conv_tol / solver.dual_obj_t
        if infres_pr <= solver.tol_feas
            solver.verbose && println("primal infeasibility detected; terminating")
            solver.status = :PrimalInfeasible
            return true
        end
    end
    if solver.primal_obj_t < 0.0
        infres_du = -max(solver.y_norm_res_t * solver.y_conv_tol, solver.z_norm_res_t * solver.z_conv_tol) / solver.primal_obj_t
        if infres_du <= solver.tol_feas
            solver.verbose && println("dual infeasibility detected; terminating")
            solver.status = :DualInfeasible
            return true
        end
    end
    if solver.mu <= solver.tol_feas * 1e-2 && solver.tau <= solver.tol_feas * 1e-2 * min(1.0, solver.kap)
        solver.verbose && println("ill-posedness detected; terminating")
        solver.status = :IllPosed
        return true
    end

    max_improve = 0.0
    for (curr, prev) in ((solver.gap, solver.prev_gap), (solver.rel_gap, solver.prev_rel_gap),
        (solver.x_feas, solver.prev_x_feas), (solver.y_feas, solver.prev_y_feas), (solver.z_feas, solver.prev_z_feas))
        if isnan(prev) || isnan(curr)
            continue
        end
        max_improve = max(max_improve, (prev - curr) / (abs(prev) + 1e-8))
    end
    if max_improve < solver.tol_slow
        if solver.slow_prev
            solver.verbose && println("slow progress in consecutive iterations; terminating")
            solver.status = :SlowProgress
            return true
        else
            solver.slow_prev = true
        end
    else
        solver.slow_prev = false
    end

    return false
end

function calc_residual(solver::HSDSolver)
    model = solver.model
    point = solver.point

    # x_residual = -A'*y - G'*z - c*tau
    x_residual = solver.x_residual
    mul!(x_residual, model.G', point.z)
    x_residual .= -model.A' * point.y - x_residual # TODO remove allocs
    solver.x_norm_res_t = norm(x_residual)
    @. x_residual -= model.c * solver.tau
    solver.x_norm_res = norm(x_residual) / solver.tau

    # y_residual = A*x - b*tau
    y_residual = solver.y_residual
    mul!(y_residual, model.A, point.x)
    solver.y_norm_res_t = norm(y_residual)
    @. y_residual -= model.b * solver.tau
    solver.y_norm_res = norm(y_residual) / solver.tau

    # z_residual = s + G*x - h*tau
    z_residual = solver.z_residual
    mul!(z_residual, model.G, point.x)
    @. z_residual += point.s
    solver.z_norm_res_t = norm(z_residual)
    @. z_residual -= model.h * solver.tau
    solver.z_norm_res = norm(z_residual) / solver.tau

    return
end

function calc_mu(solver::HSDSolver)
    solver.mu = (dot(solver.point.z, solver.point.s) + solver.tau * solver.kap) /
        (1.0 + solver.model.nu)
    return solver.mu
end
