using Plots; pyplot()
using LaTeXStrings
using Statistics

function plot_res_1x1(models, obj_func, X, Y, domain_lb, domain_ub;
    utils=nothing,
    util_opts=nothing,
    points=200,
    title="",
    show_plot=true,
    xaxis=:identity,
    yaxis=:identity,
    obj_func_label=nothing,
    model_labels=nothing,
    util_labels=nothing,
    xaxis_label=nothing,
    yaxis_util_label="normalized utility",
    yaxis_data_label=nothing,
    init_data=0,
    obj_color=nothing,
    model_colors=nothing,
    util_colors=nothing,
    kwargs...
)
    x_data = X
    y_data = Y
    xs = LinRange(domain_lb, domain_ub, points)

    # UTIL PLOT
    util_plot = nothing

    if !isnothing(utils)
        util_plot = Plots.plot(; xaxis, yaxis=(:log, [0.1, 1.]), ylabel=yaxis_util_label, kwargs...)

        for ui in 1:length(utils)
            isnothing(utils[ui]) && continue
            u_vals = utils[ui].(xs)
            if !isnothing(util_opts)
                u_opt = util_opts[ui]
                if !isnothing(u_opt)
                    u_max = max(u_opt[2], maximum(u_vals))
                    (u_max != 0) && (u_vals .*= (1/u_max) * (9/10))
                    u_vals .+= 0.1
                    scatter!(util_plot, u_opt[1], [u_opt[2] / u_max]; label="", color=util_colors[ui])
                end
            end
            label = isnothing(util_labels) ? "utility_$ui" : util_labels[ui]
            if isnothing(util_colors)
                plot!(util_plot, reduce(vcat, xs), reduce(vcat, u_vals); label)
            else
                plot!(util_plot, reduce(vcat, xs), reduce(vcat, u_vals); label, color=util_colors[ui])
            end
        end
    end

    # DATA PLOT
    data_plot = Plots.plot(; xaxis, yaxis, title, ylabel=yaxis_data_label, kwargs...)

    # obj func and data
    label = isnothing(obj_func_label) ? "obj_func" : obj_func_label
    plot!(data_plot, x->obj_func([x])[1], domain_lb[1], domain_ub[1]; label, color=(isnothing(obj_color) ? :green : obj_color))
    if init_data > 0
        scatter!(data_plot, x_data[1:init_data], y_data[1:init_data]; label="initial data", color=:yellow)
        scatter!(data_plot, x_data[init_data+1:end], y_data[init_data+1:end]; label="requested data", color=:brown)
    else
        scatter!(data_plot, x_data, y_data; label="data", color=:orange)
    end

    # models
    for i in 1:length(models)
        isnothing(models[i][1]) && continue
        x_range = (xaxis == :log) ? log_range(domain_lb[1],domain_ub[1],points) : collect(LinRange([domain_lb[1]],[domain_ub[1]],points))
        y_range = reduce(vcat, models[i][1].(x_range))
        var_range = isnothing(models[i][2]) ? nothing : reduce(vcat, models[i][2].(x_range))
        label = isnothing(model_labels) ? (length(models) > 1 ? "model_$i" : "model") : model_labels[i]
        color = isnothing(model_colors) ? :red : model_colors[i]
        plot!(data_plot, reduce(vcat, x_range), reduce(vcat, y_range); label, ribbon=var_range, points, color)
    end
    
    # COMBINED PLOT
    if isnothing(util_plot)
        p = Plots.plot(data_plot; layout=(1,1), legend=:outerright, minorgrid=true, xlabel=xaxis_label, kwargs...)
    else
        p = Plots.plot(data_plot, util_plot; layout=(2,1), legend=:outerright, minorgrid=true, xlabel=xaxis_label, kwargs...)
    end
    show_plot && display(p)
    return p
end

function plot_res_2x1(model, obj_func; points=200)
    x_range = range(A[1], B[1]; length=points)
    y_range = range(A[2], B[2]; length=points)

    surface(x_range, y_range, (x,y)->obj_func([x,y])[1]; color=:blues, alpha=0.4)
    surface!(x_range, y_range, (x,y)->model([x,y])[1]; color=:reds, alpha=0.4)
end

function plot_bsf_boxplots(results; show_plot=true, labels=nothing)
    p = plot(; title="Best-so-far solutions found\n(medians with 1st and 3rd quartiles)", legend=:topleft)

    for i in 1:length(results)
        bsf_data = getfield.(results[i], :bsf)
        bsf_data = reduce(hcat, bsf_data)
        label = isnothing(labels) ? nothing : labels[i]
        y_vals = median.(eachrow(bsf_data))
        y_err_l = y_vals .- quantile.(eachrow(bsf_data), 0.25)
        y_err_u = quantile.(eachrow(bsf_data), 0.75) .- y_vals
        plot!(p, y_vals; yerror=(y_err_l, y_err_u), label, markerstrokecolor=:auto)
    end

    show_plot && display(p)
    return p
end
