using CSV
using DataFrames
using Plots

MAX_TIME = 1800
MAX_ITER = 250

nickname = "various"

bench_file = joinpath("bench2", "various", "bench_" * nickname * ".csv")

function shifted_geomean_all(metric, conv; shift = 0, cap = Inf)
    x = copy(metric)
    x[.!conv] .= cap
    return exp(sum(log, x .+ shift) / length(x))
end
shifted_geomean_conv(metric, conv; shift = 0) = exp(sum(log, metric[conv] .+ shift) / count(conv))
shifted_geomean_all_conv(metric, all_conv; shift = 0) = exp(sum(log, metric[all_conv] .+ shift) / count(all_conv))

# comb_df = filter(:stepper => isequal("Hypatia.Solvers.CombinedStepper{Float64}"), dropmissing(all_df))
# pc_df = filter(t -> t.stepper == "Hypatia.Solvers.PredOrCentStepper{Float64}" && t.use_curve_search == true, dropmissing(all_df))
# comb_times = (comb_df[!, :solve_time])
# pc_times = (pc_df[!, :solve_time])
#
# plot(comb_times, seriestype=:stephist)
# plot!(pc_times, seriestype=:stephist)

# missig instances
# all_df = CSV.read(bench_file, DataFrame)
# insts1 = filter(t -> t.stepper == "Hypatia.Solvers.PredOrCentStepper{Float64}" && t.use_curve_search == true, dropmissing(all_df))
# insts2 = filter(t -> t.stepper == "Hypatia.Solvers.CombinedStepper{Float64}" && t.shift == 1, dropmissing(all_df))
# @show setdiff(unique(insts2[!, :inst_data]), unique(insts1[!, :inst_data]))

# boxplots
# using StatsPlots
# all_df = CSV.read(bench_file, DataFrame)
# transform!(all_df, [:inst_data, :extender] => ((x, y) -> x .* y) => :inst_key)
# all_df = combine(groupby(all_df, :inst_key), names(all_df), :status => (x -> all(isequal.("Optimal", x)) || all(isequal.("Infeasible", x))) => :all_conv)
# filter!(t -> t.all_conv == true, all_df)
# all_df = combine(groupby(all_df, :inst_key), names(all_df), :solve_time => sum => :time_sum)
# sort!(all_df, order(:time_sum, rev = true))
# all_df = all_df[1:(4 * 20), :]
# select!(all_df,
#     [:inst_data, :extender] => ((x, y) -> x .* y) => :k,
#     [:stepper, :use_corr, :use_curve_search] => ((a, b, c) -> a .* "_" .* string.(b) .* "_" .* string.(c)) => :stepper,
#     :solve_time => ByRow(log10) => :log_time,
#     # :solve_time,
#     )
# timings = unstack(all_df, :stepper, :log_time)
# # timings = unstack(all_df, :stepper, :solve_time)
# boxplot(["pc_00" "pc_01" "pc_11" "comb"], Matrix(timings[:, 2:end]), leg = false)

function post_process()
    all_df = CSV.read(bench_file, DataFrame)
    transform!(all_df,
        :status => ByRow(x -> !ismissing(x) && x in ["Optimal", "Infeasible"]) => :conv,
        [:inst_data, :extender] => ((x, y) -> x .* y) => :inst_key,
        )
    all_df = combine(groupby(all_df, :inst_key), names(all_df), :status => (x -> all(isequal.("Optimal", x)) || all(isequal.("Infeasible", x))) => :all_conv)
    filter!(t -> t.inst_set == "various", all_df)
    return all_df
end


# aggregate stuff
function agg_stats()
    output_folder = mkpath(joinpath(@__DIR__, "results"))

    all_df = post_process()

    df_agg = combine(groupby(all_df, [:stepper, :use_corr, :use_curve_search, :shift]),
        [:solve_time, :conv] => shifted_geomean_conv => :time_geomean_thisconv,
        [:iters, :conv] => shifted_geomean_conv => :iters_geomean_thisconv,
        [:solve_time, :all_conv] => shifted_geomean_all_conv => :time_geomean_allconv,
        [:iters, :all_conv] => shifted_geomean_all_conv => :iters_geomean_allconv,
        [:solve_time, :conv] => ((x, y) -> shifted_geomean_all(x, y, cap = MAX_TIME)) => :time_geomean_all,
        [:iters, :conv] => ((x, y) -> shifted_geomean_all(x, y, cap = MAX_ITER)) => :iters_geomean_all,
        :status => (x -> count(isequal("Optimal"), x)) => :optimal,
        :status => (x -> count(isequal("Infeasible"), x)) => :infeasible,
        :status => (x -> count(isequal("NumericalFailure"), x)) => :numerical,
        :status => (x -> count(isequal("SlowProgress"), x)) => :slowprogress,
        :status => (x -> count(isequal("TimeLimit"), x)) => :timelimit,
        :status => (x -> count(isequal("IterationLimit"), x)) => :iterationlimit,
        :status => length => :total,
        )
    sort!(df_agg, [order(:stepper, rev = true), :use_corr, :use_curve_search, :shift])
    CSV.write(joinpath(output_folder, "df_agg_" * nickname * ".csv"), df_agg)

    return
end
agg_stats()

# performance profiles, currently hardcoded for corrector vs no corrector
function perf_prof()
    # feature = :use_corr
    feature = :stepper
    metric = :solve_time
    # metric = :iters
    # s1 = "TRUE"
    # s2 = "FALSE"
    s1 = "Hypatia.Solvers.PredOrCentStepper{Float64}"
    s2 = "Hypatia.Solvers.CombinedStepper{Float64}"
    # s1 = true
    # s2 = false

    all_df = post_process()
    filter!(t ->
        # t -> t.stepper == "Hypatia.Solvers.PredOrCentStepper{Float64}" && t.use_curve_search == false,
        t.use_corr == true &&
        t.use_curve_search == true &&
        t.all_conv == true, # only include instances where all steppers converged (currently includes steppers not in the plot)
        all_df,
        )
    select!(all_df,
        :inst_key,
        feature,
        :solve_time => ByRow(x -> (ismissing(x) ? MAX_TIME : min(x, MAX_TIME))) => :solve_time,
        :iters => ByRow(x -> (ismissing(x) ? MAX_TIME : min(x, MAX_ITER))) => :iters,
        )
    all_df = combine(groupby(all_df, :inst_key), names(all_df), metric => (x -> x ./ minimum(x)) => :ratios)
    # transform!(all_df, metric => (x -> x ./ minimum(x)) => :ratios)
    sort!(all_df, :ratios)

    nsolvers = 2
    npts = nrow(all_df)

    # for metric in [:solve_time, :iters]
    plot(xlim = (0, log10(maximum(all_df[!, :ratios])) + 1), ylim = (0, 1))
    subdf = all_df[all_df[!, feature] .== s1, :]
    plot!(log10.(all_df[!, :ratios]), [sum(subdf[!, :ratios] .<= ti) ./ npts * nsolvers for ti in all_df[!, :ratios]], label = s1, t = :steppre)

    subdf = all_df[all_df[!, feature] .== s2, :]
    plot!(log10.(all_df[!, :ratios]), [sum(subdf[!, :ratios] .<= ti) ./ npts * nsolvers for ti in all_df[!, :ratios]], label = s2, t = :steppre)
    xaxis!("logratio")
    title!(string(feature) * " / " * string(metric) * " pp")

    return
end
# perf_prof()

# anything else that can get plotted in latex
function make_csv()
    all_df = CSV.read(bench_file, DataFrame)
    select!(all_df, :use_corr, :stepper, :solve_time, :iters)
    CSV.write(joinpath(output_folder, "df_long.csv"), df_agg)
    return
end
