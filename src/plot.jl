const MODEL_COLOR = :red
const MODEL_SAMPLE_COLOR = :orange
const F_COLOR = :green
const DATA_COLOR = :yellow
const CONSTRAINT_COLOR = :black
const CONSTRAINT_STYLE = :dash
const ACQUISITION_COLOR = :blue

"""
    PlotOptions(Plots; kwargs...)

If `PlotOptions` is passed to `BossOptions` as `callback`, the state of the optimization problem
is plotted in each iteration. Only works with one-dimensional `x` domains
but supports multi-dimensional `y`.

# Arguments
- `Plots::Module`: Evaluate `using Plots` and pass the `Plots` module to `PlotsOptions`.

# Keywords
- `f_true::Union{Nothing, Function}`: The true objective function to be plotted.
- `points::Int`: The number of points in each plotted function.
- `xaxis::Symbol`: Used to change the x axis scale (`:identity`, `:log`).
- `yaxis::Symbol`: Used to change the y axis scale (`:identity`, `:log`).
- `title::String`: The plot title.
"""
mutable struct PlotCallback{
    F<:Union{Nothing, Function},
} <: BossCallback
    Plots::Module
    prev_state::Union{Nothing, BossProblem}
    f_true::F
    points::Int
    xaxis::Symbol
    yaxis::Symbol
    title::String
end
PlotCallback(Plots::Module;
    f_true = nothing,
    points = 200,
    xaxis = :identity,
    yaxis = :identity,
    title = "BOSS optimization problem",
) = PlotCallback(Plots, nothing, f_true, points, xaxis, yaxis, title)

function (plt::PlotCallback)(problem::BossProblem;
    options::BossOptions,
    first::Bool,
    kwargs...
)
    if first
        plt.prev_state = deepcopy(problem)
    else
        acq_opt = problem.data.X[:,end]
        make_plot(plt, plt.prev_state, acq_opt; info=options.info)
        plt.prev_state = deepcopy(problem)
    end
end

function make_plot(opt::PlotCallback, problem::BossProblem, acq_opt::AbstractArray{<:Real}; info::Bool)
    info && @info "Plotting ..."
    acq = construct_acquisition(problem, BossOptions())
    display(plot_problem(opt, problem, acq, acq_opt))
end

function plot_problem(opt::PlotCallback, problem::BossProblem, acq, acq_opt)
    @assert x_dim(problem) == 1

    subplots = opt.Plots.Plot[]
    push!(subplots, [plot_y_slice(opt, problem, dim) for dim in 1:y_dim(problem)]...)
    push!(subplots, plot_acquisition(opt, problem, acq, acq_opt))
    
    opt.Plots.plot!(first(subplots); title=opt.title)
    opt.Plots.plot!(last(subplots); xlabel="x")
    
    # # Cleaner option, but subplots are not aligned on the x axis.
    # p = opt.Plots.plot(subplots...; layout=(length(subplots), 1), legend=:outerright, minorgrid=true)
    
    # Hacky option. Subplots are aligned on x axis by sharing one legend.
    for sp in subplots
        opt.Plots.plot!(sp; legend=false, minorgrid=true)
    end
    heights = vcat([1. for _ in 1:length(subplots)], 0.4)
    heights ./= sum(heights)
    layout = opt.Plots.grid(length(subplots)+1, 1; heights)
    p = opt.Plots.plot(subplots..., plot_legend(opt); layout)

    return p
end

function plot_legend(opt::PlotCallback)
    p = opt.Plots.plot(; legend=:inside, legend_column=3)
    opt.Plots.plot!(p, 1:3; xlim=(4,5), framestyle=:none, color=F_COLOR, label="f")
    opt.Plots.plot!(p, 1:3; xlim=(4,5), framestyle=:none, color=MODEL_COLOR, label="model")
    opt.Plots.scatter!(p, 1:3; xlim=(4,5), framestyle=:none, color=DATA_COLOR, label="data")
    opt.Plots.plot!(p, 1:3; xlim=(4,5), framestyle=:none, color=CONSTRAINT_COLOR, linestyle=CONSTRAINT_STYLE, label="constraints")
    opt.Plots.plot!(p, 1:3; xlim=(4,5), framestyle=:none, color=ACQUISITION_COLOR, label="acquisition")
    opt.Plots.scatter!(p, 1:3; xlim=(4,5), framestyle=:none, color=ACQUISITION_COLOR, label="optimum")
    return p
end

```
Create a plot of a single `y` dimension containing the gathered data, objective function,
constraints on `y` and the fitted model.
```
function plot_y_slice(opt::PlotCallback, problem::BossProblem, dim::Int)
    @assert x_dim(problem) == 1
    lb, ub = first.(problem.domain.bounds)

    p = opt.Plots.plot(; ylabel="y$dim", xaxis=opt.xaxis, yaxis=opt.yaxis)
    ylims = Inf, -Inf

    x_points = (opt.xaxis == :log) ? log_range(lb, ub, opt.points) : collect(LinRange(lb, ub, opt.points))

    # model
    if !isnothing(problem.params)
        if problem.params isa UniFittedParams
            # MAP -> best fit
            predict = model_posterior(problem)
            y_points = (x->predict([x])[1][dim]).(x_points)
            std_points = (x->predict([x])[2][dim]).(x_points)
            opt.Plots.plot!(p, x_points, y_points; ribbon=std_points, label="model", color=MODEL_COLOR)
            ylims = update_ylims(ylims, y_points)
        
        else # problem.params isa MultiFittedParams
            # BI -> samples & mean
            predicts = model_posterior(problem)
            for i in eachindex(predicts)
                y_points = (x->predicts[i]([x])[1][dim]).(x_points)
                # std_points = (x->predicts[i]([x])[2][dim]).(x_points)
                label = (i == 1) ? "model samples" : nothing
                opt.Plots.plot!(p, x_points, y_points; label, color=MODEL_SAMPLE_COLOR, style=:dash, alpha=0.2)
            end

            pred_mean(x) = mean(map(p->p(x)[1][dim], predicts))
            y_points = (x->first(pred_mean([x]))).(x_points)
            opt.Plots.plot!(p, x_points, y_points; label="averaged model", color=MODEL_COLOR)
            ylims = update_ylims(ylims, y_points)
        end
    end

    # constraint
    if !isinf(problem.y_max[dim])
        opt.Plots.plot!(p, x->problem.y_max[dim], lb, ub; label="constraint", color=CONSTRAINT_COLOR, linestyle=CONSTRAINT_STYLE, thickness_scaling=3, points=opt.points)
    end

    # f
    if !isnothing(opt.f_true)
        f_slice = x->opt.f_true([x])[dim]
        y_points = f_slice.(x_points)
        opt.Plots.plot!(p, x_points, y_points; label="f", color=F_COLOR)
        ylims = update_ylims(ylims, y_points)
    end

    # data
    if !isempty(problem.data)
        opt.Plots.scatter!(p, vec(problem.data.X), vec(problem.data.Y[dim,:]); label="data", color=DATA_COLOR, markersize=2.)
    end

    add = maximum(abs.(ylims)) / 10.
    ylims = ylims[1]-add, ylims[2]+add
    opt.Plots.plot!(p; ylims)
    return p
end

```
Create a plot of the acquisition function.
```
function plot_acquisition(opt::PlotCallback, problem::BossProblem, acq, acq_opt)
    @assert x_dim(problem) == 1
    lb, ub = first.(problem.domain.bounds)

    p = opt.Plots.plot(; ylabel="acquisition", xaxis=opt.xaxis, yaxis=opt.yaxis)

    acq_(x) = in_domain(x, problem.domain) ? acq(x) : 0.
    x_points = (opt.xaxis == :log) ? log_range(lb, ub, opt.points) : collect(LinRange(lb, ub, opt.points))
    y_points = (x->acq_([x])).(x_points)
    opt.Plots.plot!(p, x_points, y_points; label="acquisition", color=ACQUISITION_COLOR)

    if !isnothing(acq_opt)
        opts = eachcol(acq_opt) |> collect
        vals = acq_.(opts)
        opt.Plots.scatter!(p, opts, vals; label="optimum", color=ACQUISITION_COLOR)
    end

    return p
end

function update_ylims(ylims, y_points)
    ymin, ymax = minimum(y_points), maximum(y_points)
    min(ymin, ylims[1]), max(ymax, ylims[2])
end

```
Return points distributed evenly over a given logarithmic range.
```
function log_range(a, b, len)
    a = log10.(a)
    b = log10.(b)
    range = collect(LinRange(a, b, len))
    range = [10 .^ range[i] for i in 1:len]
    return range
end
