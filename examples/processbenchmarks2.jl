using CSV
using DataFrames
using Plots
using Printf
using BenchmarkProfiles

enhancements = ["basic", "TOA", "curve", "comb", "shift"]
process_entry(x::Float64) = @sprintf("%.2f", x)
process_entry(x::Int) = string(x)

bench_file = joinpath("bench2", "various", "bench" * ".csv")
output_folder = mkpath(joinpath(@__DIR__, "results"))
tex_folder = mkpath(joinpath(output_folder, "tex"))
raw_folder = mkpath(joinpath(output_folder, "raw"))

function shifted_geomean(metric::AbstractVector, conv::AbstractVector{Bool}; shift = 0, cap = Inf, skipnotfinite = false, use_cap = false)
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

function post_process()
    all_df = CSV.read(bench_file, DataFrame)
    transform!(all_df,
        :status => ByRow(x -> !ismissing(x) && x in ["Optimal", "PrimalInfeasible", "DualInfeasible"]) => :conv,
        # each instance is identified by instance data + extender combination
        [:example, :inst_data, :extender] => ((x, y, z) -> x .* y .* z) => :inst_key,
        )
    # assumes that nothing returned incorrect status, which is checked manually
    all_df = combine(groupby(all_df, :inst_key), names(all_df), :status => (x -> all(in(["Optimal", "PrimalInfeasible", "DualInfeasible"]).(x))) => :every_conv)
    # remove precompile instances
    filter!(t -> t.inst_set == "various", all_df)
    return all_df
end

# aggregate stuff
function agg_stats()
    all_df = post_process()
    conv = all_df[!, :conv]
    MAX_TIME = maximum(all_df[!, :solve_time][conv])
    MAX_ITER = maximum(all_df[!, :iters][conv])
    time_shift = 0

    df_agg = combine(groupby(all_df, [:stepper, :toa, :curve, :shift]),
        [:solve_time, :conv] => ((x, y) -> shifted_geomean(x, y, shift = time_shift)) => :time_geomean_thisconv,
        [:iters, :conv] => ((x, y) -> shifted_geomean(x, y, shift = 1)) => :iters_geomean_thisconv,
        [:solve_time, :every_conv] => ((x, y) -> shifted_geomean(x, y, shift = time_shift)) => :time_geomean_everyconv,
        [:iters, :every_conv] => ((x, y) -> shifted_geomean(x, y, shift = 1)) => :iters_geomean_everyconv,
        [:solve_time, :conv] => ((x, y) -> shifted_geomean(x, y, cap = MAX_TIME, use_cap = true, shift = time_shift)) => :time_geomean_all,
        [:iters, :conv] => ((x, y) -> shifted_geomean(x, y, shift = 1, cap = MAX_ITER, use_cap = true)) => :iters_geomean_all,
        :status => (x -> count(isequal("Optimal"), x)) => :optimal,
        :status => (x -> count(isequal("PrimalInfeasible"), x)) => :priminfeas,
        :status => (x -> count(isequal("DualInfeasible"), x)) => :dualinfeas,
        :status => (x -> count(isequal("NumericalFailure"), x)) => :numerical,
        :status => (x -> count(isequal("SlowProgress"), x)) => :slowprogress,
        :status => (x -> count(isequal("TimeLimit"), x)) => :timelimit,
        :status => (x -> count(isequal("IterationLimit"), x)) => :iterationlimit,
        :status => length => :total,
        )
    sort!(df_agg, [order(:stepper, rev = true), :toa, :curve, :shift])
    CSV.write(joinpath(raw_folder, "agg" * ".csv"), df_agg)

    # combine feasible and infeasible statuses
    transform!(df_agg, [:optimal, :priminfeas, :dualinfeas] => ByRow((x...) -> sum(x)) => :converged)
    cols = [:converged, :iters_geomean_thisconv, :iters_geomean_everyconv, :iters_geomean_all, :time_geomean_thisconv, :time_geomean_everyconv, :time_geomean_all]
    sep = " & "
    tex = open(joinpath(tex_folder, "agg" * ".tex"), "w")
    for i in 1:length(enhancements)
        row_str = enhancements[i]
        for c in cols
            subdf = df_agg[!, c]
            row_str *= sep * process_entry(subdf[i])
        end
        row_str *= " \\\\"
        println(tex, row_str)
    end
    close(tex)

    return
end
agg_stats()

function subtime()
    total_shift = 1e-4
    piter_shift = 1e-4
    all_df = post_process()
    divfunc(x, y) = (x ./ y)
    preproc_cols = [:time_rescale, :time_initx, :time_inity, :time_unproc]
    transform!(all_df,
        [:time_upsys, :iters] => divfunc => :time_upsys_piter,
        [:time_uprhs, :iters] => divfunc => :time_uprhs_piter,
        [:time_getdir, :iters] => divfunc => :time_getdir_piter,
        [:time_search, :iters] => divfunc => :time_search_piter,
        preproc_cols => ((x...) -> sum(x)) => :time_linalg,
        )

    metrics = [:linalg, :uplhs, :uprhs, :getdir, :search, :uplhs_piter, :uprhs_piter, :getdir_piter, :search_piter]
    sets = [:_thisconv, :_everyconv, :_all]

    # get values to replace unconverged instances for the "all" group
    conv = all_df[!, :conv]
    max_linalg = maximum(all_df[!, :time_linalg][conv])
    max_upsys = maximum(all_df[!, :time_upsys][conv])
    max_uprhs = maximum(all_df[!, :time_uprhs][conv])
    max_getdir = maximum(all_df[!, :time_getdir][conv])
    max_search = maximum(all_df[!, :time_search][conv])
    max_upsys_iter = maximum(all_df[!, :time_upsys_piter][conv])
    max_uprhs_iter = maximum(all_df[!, :time_uprhs_piter][conv])
    max_getdir_iter = maximum(all_df[!, :time_getdir_piter][conv])
    max_search_iter = maximum(all_df[!, :time_search_piter][conv])

    function get_subtime_df(set, convcol, use_cap)
        subtime_df = combine(groupby(all_df, [:stepper, :toa, :curve, :shift]),
            [:time_linalg, convcol] => ((x, y) -> shifted_geomean(x, y, shift = total_shift, cap = max_linalg, use_cap = use_cap)) => Symbol(:linalg, set),
            [:time_upsys, convcol] => ((x, y) -> shifted_geomean(x, y, shift = total_shift, cap = max_upsys, use_cap = use_cap)) => Symbol(:uplhs, set),
            [:time_uprhs, convcol] => ((x, y) -> shifted_geomean(x, y, shift = total_shift, cap = max_uprhs, use_cap = use_cap)) => Symbol(:uprhs, set),
            [:time_getdir, convcol] => ((x, y) -> shifted_geomean(x, y, shift = total_shift, cap = max_getdir, use_cap = use_cap)) => Symbol(:getdir, set),
            [:time_search, convcol] => ((x, y) -> shifted_geomean(x, y, shift = total_shift, cap = max_search, use_cap = use_cap)) => Symbol(:search, set),
            [:time_upsys_piter, convcol] => ((x, y) -> shifted_geomean(x, y, shift = piter_shift, skipnotfinite = true, cap = max_upsys_iter, use_cap = use_cap)) => Symbol(:uplhs_piter, set),
            [:time_uprhs_piter, convcol] => ((x, y) -> shifted_geomean(x, y, shift = piter_shift, skipnotfinite = true, cap = max_uprhs_iter, use_cap = use_cap)) => Symbol(:uprhs_piter, set),
            [:time_getdir_piter, convcol] => ((x, y) -> shifted_geomean(x, y, shift = piter_shift, skipnotfinite = true, cap = max_getdir_iter, use_cap = use_cap)) => Symbol(:getdir_piter, set),
            [:time_search_piter, convcol] => ((x, y) -> shifted_geomean(x, y, shift = piter_shift, skipnotfinite = true, cap = max_search_iter, use_cap = use_cap)) => Symbol(:search_piter, set),
            )
        sort!(subtime_df, [order(:stepper, rev = true), :toa, :curve, :shift])
        CSV.write(joinpath(raw_folder, "subtime" * string(set) * "_.csv"), subtime_df)
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

        subtime_tex = open(joinpath(tex_folder, "subtime" * string(s) * ".tex"), "w")
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
subtime()

# performance profiles, currently hardcoded for corrector vs no corrector
function perf_prof(; feature = :stepper, metric = :solve_time)
    if feature == :stepper
        s1 = "PredOrCentStepper"
        s2 = "CombinedStepper"
        toa = [true]
        curve = [true]
        stepper = [s1, s2]
        shift = [0]
    elseif feature == :shift
        s1 = 0
        s2 = 2
        stepper = ["CombinedStepper"]
        toa = [true]
        curve = [true]
        shift = [s1, s2]
    else
        s1 = false
        s2 = true
        stepper = ["PredOrCentStepper"]
        shift = [0]
        if feature == :toa
            curve = [false]
            toa = [s1, s2]
        elseif feature == :curve
            toa = [true]
            curve = [s1, s2]
        end
    end

    all_df = post_process()
    filter!(t ->
        t.stepper in stepper &&
        t.toa in toa &&
        t.curve in curve &&
        t.shift in shift,
        all_df
        )

    # remove instances where neither stepper being compared converged
    # all_df = combine(groupby(all_df, :inst_key), names(all_df), :conv => any => :any_conv)
    # filter!(t -> t.any_conv, all_df)

    # BenchmarkProfiles expects NaNs for failures
    select!(all_df,
        :inst_key,
        feature,
        [metric, :conv] => ByRow((x, y) -> (y ? x : NaN)) => metric,
        )

    wide_df = unstack(all_df, feature, metric)
    (ratios, max_ratio) = BenchmarkProfiles.performance_ratios(Matrix{Float64}(wide_df[!, string.([s1, s2])]))

    (x_plot, y_plot, max_ratio) = BenchmarkProfiles.performance_profile_data(Matrix{Float64}(wide_df[!, string.([s1, s2])]), logscale = true)
    for s = 1 : 2
        x = vcat(0, repeat(x_plot[s], inner = 2))
        y = vcat(0, 0, repeat(y_plot[s][1:(end - 1)], inner = 2), y_plot[s][end])
        CSV.write(joinpath(output_folder, string(feature) * "_" * string(metric) * "_$(s)" * "_pp" * ".csv"), DataFrame(x = x, y = y))
    end
    return
end

for feature in [:stepper, :curve, :toa, :shift], metric in [:solve_time, :iters]
    @show feature, metric
    perf_prof(feature = feature, metric = metric)
end

function instancestats()
    all_df = post_process()

    one_solver = filter!(t ->
        t.stepper == "PredOrCentStepper" &&
        t.toa == 0 &&
        t.curve == 0,
        all_df
        )
    inst_df = select(one_solver, :num_cones => ByRow(log10) => :numcones, [:n, :p, :q] => ((x, y, z) -> log10.(x .+ y .+ z)) => :npq)
    CSV.write(joinpath(output_folder, "inststats.csv"), inst_df)
    # for solve times, only include converged instances
    solve_times = filter!(t -> t.conv, one_solver)
    CSV.write(joinpath(output_folder, "solvetimes.csv"), select(solve_times, :solve_time => ByRow(log10) => :time))

    # only used to get list of cones manually
    ex_df = combine(groupby(one_solver, :example),
        :cone_types => (x -> union(eval.(Meta.parse.(x)))) => :cones,
        :cone_types => length => :num_instances,
        )
    CSV.write(joinpath(raw_folder, "examplestats.csv"), ex_df)

    return
end
instancestats()

# comb_df = filter(:stepper => isequal("CombinedStepper"), dropmissing(all_df))
# pc_df = filter(t -> t.stepper == "PredOrCentStepper" && t.curve == true, dropmissing(all_df))
# comb_times = (comb_df[!, :solve_time])
# pc_times = (pc_df[!, :solve_time])
#
# plot(comb_times, seriestype=:stephist)
# plot!(pc_times, seriestype=:stephist)

# missig instances
# all_df = CSV.read(bench_file, DataFrame)
# insts1 = filter(t -> t.stepper == "PredOrCentStepper" && t.curve == true, dropmissing(all_df))
# insts2 = filter(t -> t.stepper == "CombinedStepper" && t.shift == 1, dropmissing(all_df))
# @show setdiff(unique(insts2[!, :inst_data]), unique(insts1[!, :inst_data]))

# # boxplots
# using StatsPlots
# all_df = post_process()
# transform!(all_df, [:inst_data, :extender] => ((x, y) -> x .* y) => :inst_key)
# all_df = combine(groupby(all_df, :inst_key), names(all_df), :status => (x -> all(isequal.("Optimal", x)) || all(isequal.("PrimalInfeasible", x))) => :every_conv)
# filter!(t -> t.every_conv, all_df)
# filter!(t -> t.shift == 0, all_df)
# select!(all_df,
#     [:inst_data, :extender] => ((x, y) -> x .* y) => :k,
#     [:stepper, :toa, :curve] => ((a, b, c) -> a .* "_" .* string.(b) .* "_" .* string.(c)) => :stepper,
#     :solve_time => ByRow(log10) => :log_time,
#     )
# timings = unstack(all_df, :stepper, :log_time)
# boxplot(["pc_00" "pc_01" "pc_11" "comb"], Matrix(timings[:, 2:end]), leg = false)

# function stats_plots()
#     all_df = post_process()
#     histogram(log10.(all_df[!, :solve_time]))
#     title!("log10 solve time")
#     png("solvehist")
#
#     histogram(log10.(sum(eachcol(all_df[!, [:n, :p, :q]]))))
#     title!("log10 n + p + q")
#     png("npqhist")
#
#     histogram(log10.(max.(0.01, all_df[!, :n] - all_df[!, :p])))
#     title!("log10(n - p)")
#     png("nphist")
#
#     histogram(all_df[!, :solve_time])
#     title!("num cones")
#     png("Khist")
# end
# stats_plots()
