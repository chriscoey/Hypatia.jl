using CSV
using DataFrames
using Printf
using BenchmarkProfiles

enhancements = ["basic", "TOA", "curve", "comb", "back"]
process_entry(x::Float64) = @sprintf("%.2f", x)
process_entry(x::Int) = string(x)

bench_file = joinpath(@__DIR__, "raw", "bench.csv")
output_dir = mkpath(joinpath(@__DIR__, "analysis"))
tex_dir = mkpath(joinpath(output_dir, "tex"))
stats_dir = mkpath(joinpath(output_dir, "stats"))
csv_dir = mkpath(joinpath(output_dir, "csvs"))

function shifted_geomean(
    metric::AbstractVector{<:Real},
    conv::AbstractVector{Bool};
    shift::Real = 0,
    cap::Real = Inf,
    skipnotfinite::Bool = false,
    use_cap::Bool = false,
    )
    if use_cap
        x = copy(metric)
        x[.!conv] .= cap
    else
        x = metric[conv]
    end
    if skipnotfinite
        x = x[isfinite.(x)]
    end
    return exp(sum(log, x .+ shift) / length(x)) - shift
end

function preprocess_df()
    all_df = CSV.read(bench_file, DataFrame)
    transform!(all_df,
        :status => ByRow(x -> !ismissing(x) && x in
            ["Optimal", "PrimalInfeasible", "DualInfeasible"]) => :conv,
        # each instance is identified by instance data + extender combination
        [:example, :inst_data, :extender] => 
        ((x, y, z) -> x .* y .* z) => :inst_key,
        :solver_options => ByRow(x -> eval(Meta.parse(x))[1]) => :solver_options,
        )
    # assumes that nothing returned incorrect status, which is checked manually
    all_df = combine(groupby(all_df, :inst_key), names(all_df),
        :conv => all => :every_conv)
    # remove precompile instances
    filter!(t -> t.inst_set == "various", all_df)
    return all_df
end

function make_agg_tables(all_df)
    conv = all_df[!, :conv]
    max_time = maximum(all_df[!, :solve_time][conv])
    max_iter = maximum(all_df[!, :iters][conv])
    time_shift = 1e-3

    df_agg = combine(groupby(all_df, :solver_options),
        [:solve_time, :conv] => ((x, y) ->
            shifted_geomean(x, y, shift = time_shift)) => :time_geomean_thisconv,
        [:iters, :conv] => ((x, y) ->
            shifted_geomean(x, y, shift = 1)) => :iters_geomean_thisconv,
        [:solve_time, :every_conv] => ((x, y) ->
            shifted_geomean(x, y, shift = time_shift)) => :time_geomean_everyconv,
        [:iters, :every_conv] => ((x, y) ->
            shifted_geomean(x, y, shift = 1)) => :iters_geomean_everyconv,
        [:solve_time, :conv] => ((x, y) ->
            shifted_geomean(x, y, cap = max_time, use_cap = true,
            shift = time_shift)) => :time_geomean_all,
        [:iters, :conv] => ((x, y) ->
            shifted_geomean(x, y, shift = 1, cap = max_iter,
            use_cap = true)) => :iters_geomean_all,
        :status => (x -> count(isequal("Optimal"), x)) => :optimal,
        :status => (x -> count(isequal("PrimalInfeasible"), x)) => :priminfeas,
        :status => (x -> count(isequal("DualInfeasible"), x)) => :dualinfeas,
        :status => (x -> count(isequal("NumericalFailure"), x)) => :numerical,
        :status => (x -> count(isequal("SlowProgress"), x)) => :slowprogress,
        :status => (x -> count(isequal("TimeLimit"), x)) => :timelimit,
        :status => (x -> count(isequal("IterationLimit"), x)) => :iterationlimit,
        :status => length => :total,
        )

    sort!(df_agg, order(:solver_options, by = (x ->
        findfirst(isequal(x), lowercase.(enhancements)))))
    CSV.write(joinpath(stats_dir, "agg" * ".csv"), df_agg)

    # combine feasible and infeasible statuses
    transform!(df_agg, [:optimal, :priminfeas, :dualinfeas] =>
        ByRow((x...) -> sum(x)) => :converged)

    cols = [:converged, :iters_geomean_thisconv, :iters_geomean_everyconv,
        :iters_geomean_all, :time_geomean_thisconv, :time_geomean_everyconv,
        :time_geomean_all]
    sep = " & "
    tex = open(joinpath(tex_dir, "agg" * ".tex"), "w")

    for i in 1:length(enhancements)
        row_str = enhancements[i]
        for c in cols
            subdf = df_agg[!, c]
            x = (startswith(string(c), "time") ? subdf[i] * 1000 : subdf[i])
            row_str *= sep * process_entry(x)
        end
        row_str *= " \\\\"
        println(tex, row_str)
    end
    close(tex)

    return
end

function make_subtime_tables(all_df)
    total_shift = 1e-4
    piter_shift = 1e-5
    divfunc(x, y) = (x ./ y)
    preproc_cols = [:time_rescale, :time_initx, :time_inity, :time_unproc]
    transform!(all_df,
        [:time_upsys, :iters] => divfunc => :time_upsys_piter,
        [:time_uprhs, :iters] => divfunc => :time_uprhs_piter,
        [:time_getdir, :iters] => divfunc => :time_getdir_piter,
        [:time_search, :iters] => divfunc => :time_search_piter,
        preproc_cols => ((x...) -> sum(x)) => :time_linalg,
        )

    metrics = [:linalg, :uplhs, :uprhs, :getdir, :search, :uplhs_piter,
        :uprhs_piter, :getdir_piter, :search_piter]
    sets = [:_thisconv, :_everyconv, :_all]

    # get values to replace unconverged instances for the "all" group
    conv = all_df[!, :conv]
    max_linalg = maximum(all_df[!, :time_linalg][conv])
    max_upsys = maximum(all_df[!, :time_upsys][conv])
    max_uprhs = maximum(all_df[!, :time_uprhs][conv])
    max_getdir = maximum(all_df[!, :time_getdir][conv])
    max_search = maximum(all_df[!, :time_search][conv])

    # get maximum values to use as caps for the "all" subset
    skipnan(x) = (isnan(x) ? 0 : x)
    max_upsys_iter = maximum(skipnan, all_df[!, :time_upsys_piter][conv])
    max_uprhs_iter = maximum(skipnan, all_df[!, :time_uprhs_piter][conv])
    max_getdir_iter = maximum(skipnan, all_df[!, :time_getdir_piter][conv])
    max_search_iter = maximum(skipnan, all_df[!, :time_search_piter][conv])

    function get_subtime_df(set, convcol, use_cap)
        subtime_df = combine(groupby(all_df, :solver_options),
            [:time_linalg, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = total_shift, cap = max_linalg,
                use_cap = use_cap)) => Symbol(:linalg, set),
            [:time_upsys, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = total_shift, cap = max_upsys,
                use_cap = use_cap)) => Symbol(:uplhs, set),
            [:time_uprhs, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = total_shift, cap = max_uprhs,
                use_cap = use_cap)) => Symbol(:uprhs, set),
            [:time_getdir, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = total_shift, cap = max_getdir,
                use_cap = use_cap)) => Symbol(:getdir, set),
            [:time_search, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = total_shift, cap = max_search,
                use_cap = use_cap)) => Symbol(:search, set),
            [:time_upsys_piter, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = piter_shift, skipnotfinite = true,
                cap = max_upsys_iter, use_cap = use_cap)) =>
                Symbol(:uplhs_piter, set),
            [:time_uprhs_piter, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = piter_shift, skipnotfinite = true,
                cap = max_uprhs_iter, use_cap = use_cap)) =>
                Symbol(:uprhs_piter, set),
            [:time_getdir_piter, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = piter_shift, skipnotfinite = true,
                cap = max_getdir_iter, use_cap = use_cap)) =>
                Symbol(:getdir_piter, set),
            [:time_search_piter, convcol] => ((x, y) ->
                shifted_geomean(x, y, shift = piter_shift, skipnotfinite = true,
                cap = max_search_iter, use_cap = use_cap)) =>
                Symbol(:search_piter, set),
            )

        sort!(subtime_df, order(:solver_options,
            by = (x -> findfirst(isequal(x), lowercase.(enhancements)))))
        CSV.write(joinpath(stats_dir, "subtime" * string(set) * ".csv"),
            subtime_df)

        return subtime_df
    end

    sep = " & "
    for s in sets
        if s == :_thisconv
            convcol = :conv
            use_cap = false
        elseif s == :_everyconv
            convcol = :every_conv
            use_cap = false
        elseif s == :_all
            convcol = :conv
            use_cap = true
        end
        subtime_df = get_subtime_df(s, convcol, use_cap)

        subtime_tex = open(joinpath(tex_dir, "subtime" * string(s) * ".tex"), "w")
        for i in 1:nrow(subtime_df)
            row_str = sep * enhancements[i]
            for m in metrics
                col = Symbol(m, s)
                row_str *= sep * process_entry(subtime_df[i, col] * 1000)
            end
            row_str *= " \\\\"
            println(subtime_tex, row_str)
        end
        close(subtime_tex)
    end

    return
end

function make_perf_profiles(all_df, comp, metric)
    pp = filter(t -> t.solver_options in comp, all_df)
    # BenchmarkProfiles.jl expects NaNs for failures
    select!(pp,
        :inst_key,
        :solver_options,
        [metric, :conv] => ByRow((x, y) -> (y ? x : NaN)) => metric,
        )
    wide_df = unstack(pp, :solver_options, metric)
    (x_plot, y_plot, max_ratio) = BenchmarkProfiles.performance_profile_data(
        Matrix{Float64}(wide_df[!, string.(comp)]), logscale = true)

    # make steps like :steppost in Plots
    for s in 1:2
        x = vcat(0, repeat(x_plot[s], inner = 2))
        y = vcat(0, 0, repeat(y_plot[s][1:(end - 1)], inner = 2), y_plot[s][end])
        CSV.write(joinpath(csv_dir, comp[s] * "_vs_" * comp[2 - s + 1] * "_" *
            string(metric) * ".csv"), DataFrame(x = x, y = y))
    end
    return
end

function instance_stats(all_df)
    all_df = transform(all_df, [:n, :p, :q] => ((x, y, z) -> x .+ y .+ z) => :npq)
    basic_solver = filter(t -> t.solver_options == "basic", all_df)

    # get stats from basic
    CSV.write(joinpath(csv_dir, "basic.csv"), select(basic_solver,
        :num_cones => ByRow(log10) => :log_numcones,
        :npq => ByRow(log10) => :log_npq,
        ),)

    # basic and converged
    basic_solver_conv = filter(t -> t.conv, basic_solver)
    CSV.write(joinpath(csv_dir, "basicconv.csv"), select(basic_solver_conv,
        :iters,
        :solve_time,
        :solve_time => ByRow(log10) => :log_solve_time,
        :npq,
        ),)

    # back and converged
    back_solver = filter(t -> t.solver_options == "back", all_df)
    back_solver_conv = filter(t -> t.conv, back_solver)
    CSV.write(joinpath(csv_dir, "backconv.csv"), select(back_solver_conv,
        :solve_time,
        :npq,
        [:time_uprhs, :solve_time] => ((x, y) -> x ./ y) => :prop_rhs,
        ),)

    # basic and back where both converged
    two_solver = filter(t -> t.solver_options in ("basic", "back"), all_df)
    two_solver = combine(groupby(two_solver, :inst_key), names(all_df),
        :conv => all => :two_conv)
    two_solver_conv = filter(t -> t.two_conv, two_solver)
    two_solver_conv = combine(groupby(two_solver_conv, :inst_key),
        [:solver_options, :solve_time], :solve_time =>
        (x -> (x[1] - x[2]) / x[1]) => :improvement)
    filter!(t -> t.solver_options == "basic", two_solver_conv)
    CSV.write(joinpath(csv_dir, "basicbackconv.csv"),
        select(two_solver_conv, :solve_time, :improvement))

    # only used to get list of cones manually
    ex_df = combine(groupby(basic_solver, :example),
        :cone_types => (x -> union(eval.(Meta.parse.(x)))) => :cones,
        :cone_types => length => :num_instances,
        )
    CSV.write(joinpath(stats_dir, "examplestats.csv"), ex_df)

    # count instances with loosened tols
    examples_dir = "../../examples"
    include(joinpath(examples_dir, "common.jl"))
    n = 1
    for (m, l) in (("JuMP", 3), ("native", 2))
        include(joinpath(examples_dir, "common_" * m * ".jl"))
        m_df = filter(t -> (t.model_type == m), all_df)
        for ex_name in unique(m_df[!, :example])
            include(joinpath(examples_dir, ex_name, m * ".jl"))
            (_, ex_insts) = include(joinpath(examples_dir, ex_name,
                m * "_test.jl"))
            for inst in ex_insts["various"]
                (length(inst) == l) && (n += 1)
            end
        end
    end
    @show n

    return
end

function post_process()
    all_df = preprocess_df()
    make_agg_tables(all_df)
    make_subtime_tables(all_df)
    comp_list = [
        ["basic", "toa"],
        ["toa", "curve"],
        ["curve", "comb"],
        ["comb", "back"],
        ]
    for comp in comp_list, metric in [:solve_time, :iters]
        make_perf_profiles(all_df, comp, metric)
    end
    instance_stats(all_df)
    return
end
post_process()
;
