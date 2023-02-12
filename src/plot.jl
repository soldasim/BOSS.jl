function plot_problem(opt::PlotOptions, problem::OptimizationProblem)
    @assert x_dim(problem) == 1

    subplots = opt.Plots.Plot[]
    push!(subplots, [plot_y_slice(opt, problem, dim) for dim in 1:y_dim(problem)]...)
    isnothing(opt.acquisition) || push!(subplots, plot_acquisition(opt, problem))
    
    opt.Plots.plot!(first(subplots); title=opt.title)
    opt.Plots.plot!(last(subplots); xlabel="x")
    p = opt.Plots.plot(subplots...; layout=(length(subplots), 1), legend=:outerright, minorgrid=true)

    display(p)
    return p
end

function plot_y_slice(opt::PlotOptions, problem::OptimizationProblem, dim::Int)
    @assert x_dim(problem) == 1
    lb, ub = first.(get_bounds(problem.domain))

    p = opt.Plots.plot(; ylabel="y$dim", xaxis=opt.xaxis, yaxis=opt.yaxis)
    ylims = Inf, -Inf

    x_points = (opt.xaxis == :log) ? log_range(lb, ub, opt.points) : collect(LinRange(lb, ub, opt.points))

    # model
    if problem.data isa ExperimentDataPost
        if problem.data isa ExperimentDataPost{MLE}
            # MLE -> best fit
            predict = model_posterior(problem.model, problem.data)
            y_points = (x->predict([x])[1][dim]).(x_points)
            var_points = (x->predict([x])[2][dim]).(x_points)
            opt.Plots.plot!(p, x_points, y_points; ribbon=var_points, label="model", color=:red)
            ylims = update_ylims(ylims, y_points)
        
        else
            # BI -> samples & mean
            predicts = model_posterior(problem.model, problem.data)
            for i in eachindex(predicts)
                y_points = (x->first(predicts[i]([x])[1])).(x_points)
                # var_points = (x->first(predicts[i]([x])[2])).(x_points)
                label = (i == 1) ? "model samples" : nothing
                opt.Plots.plot!(p, x_points, y_points; label, color=:red, alpha=0.2)
            end

            pred_mean(x) = mean(map(p->p(x)[1], predicts))
            pred_var(x) = mean(map(p->p(x)[2], predicts))
            y_points = (x->first(pred_mean([x])[1])).(x_points)
            var_points = (x->first(pred_var([x])[2])).(x_points)
            opt.Plots.plot!(p, x_points, y_points; ribon=var_points, label="model mean", color=:red)
            ylims = update_ylims(ylims, y_points)
        end
    end

    # constraint
    if !isinf(problem.cons[dim])
        opt.Plots.plot!(p, x->problem.cons[dim], lb, ub; label="constraint", color=:black, linestyle=:dash, thickness_scaling=3, points=opt.points)
    end

    # f
    if !isnothing(opt.f_true)
        f_slice = x->opt.f_true([x])[dim]
        y_points = f_slice.(x_points)
        opt.Plots.plot!(p, x_points, y_points; label="f", color=:green)
        ylims = update_ylims(ylims, y_points)
    end

    # data
    if !isempty(problem.data)
        opt.Plots.scatter!(p, vec(problem.data.X), vec(problem.data.Y[dim,:]); label="data", color=:yellow, markersize=2.)
    end

    opt.Plots.plot!(p; ylims)
    return p
end

function plot_acquisition(opt::PlotOptions, problem::OptimizationProblem)
    @assert x_dim(problem) == 1
    lb, ub = first.(get_bounds(problem.domain))

    p = opt.Plots.plot(; ylabel="acquisition", xaxis=opt.xaxis, yaxis=opt.yaxis)

    if !isnothing(opt.acquisition)
        acq(x) = in_domain(problem.domain, [x]) ? opt.acquisition([x]) : 0.
        x_points = (opt.xaxis == :log) ? log_range(lb, ub, opt.points) : collect(LinRange(lb, ub, opt.points))
        y_points = acq.(x_points)
        opt.Plots.plot!(p, x_points, y_points; label="acquisition", color=:blue)

        if !isnothing(opt.acq_opt)
            o = first(opt.acq_opt)
            opt.Plots.scatter!(p, [o], [acq(o)]; label="optimum", color=:blue)
        end
    end

    return p
end

function update_ylims(ylims, y_points)
    ymin, ymax = minimum(y_points), maximum(y_points)
    min(ymin, ylims[1]), max(ymax, ylims[2])
end

# Return points distributed evenly over a given logarithmic range.
function log_range(a, b, len)
    a = log10.(a)
    b = log10.(b)
    range = collect(LinRange(a, b, len))
    range = [10 .^ range[i] for i in 1:len]
    return range
end
