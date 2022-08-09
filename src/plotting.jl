using Plots; pyplot()
using LaTeXStrings
using Statistics

function create_plots(f_true, utils, util_opts, models, c_model, X, Y;
    iter,
    y_dim,
    constrained,
    c_count,
    domain_lb,
    domain_ub,
    init_data_size,
    show_plots,
    kwargs...
)
    colors = [:red, :purple, :blue]
    labels = ["param", "semiparam", "nonparam"]
    util_label = constrained ? "cwEI" : "EI"
    constraints = constrained ? [x -> constraint_probabilities(c_model)(x)[i] for i in 1:c_count] : nothing

    plots = Plots.Plot[]
    for d in 1:y_dim
        title = (y_dim > 1) ? "ITER $iter, DIM $d" : "ITER $iter"
        models = model_dim_slice.(models, d)
        
        p = plot_res_1x1(models, x -> f_true(x)[d], X, Y, domain_lb, domain_ub; utils, util_opts, constraints, yaxis_constraint_label="constraint\nsatisfaction prob.", title, init_data=init_data_size, model_colors=colors, util_colors=colors, model_labels=labels, util_labels=labels, show_plot=show_plots, yaxis_util_label=util_label, kwargs...)
        push!(plots, p)
    end
    return plots
end

function plot_res_1x1(models, obj_func, X, Y, domain_lb, domain_ub;
    utils=nothing,
    util_opts=nothing,
    constraints=nothing,
    points=200,
    title="",
    show_plot=true,
    xaxis=:identity,
    yaxis=:identity,
    obj_func_label=nothing,
    model_labels=nothing,
    util_labels=nothing,
    constraint_labels=nothing,
    xaxis_label=nothing,
    yaxis_util_label="normalized utility",
    yaxis_data_label=nothing,
    yaxis_constraint_label="constraints",
    init_data=0,
    obj_color=nothing,
    model_colors=nothing,
    util_colors=nothing,
    constraint_colors=nothing,
    kwargs...
)
    x_data = X
    y_data = Y
    xs = LinRange(domain_lb, domain_ub, points)

    # UTIL PLOT
    util_plot = nothing

    if !isnothing(utils)
        util_plot = Plots.plot(; xaxis, yaxis=(:log, [0.1, 1.]), ylabel=yaxis_util_label, kwargs...)

        for ui in 1:lastindex(utils)
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

    # CONSTRAINTS PLOT
    constraint_plot = nothing

    if !isnothing(constraints)
        constraint_plot = Plots.plot(; xaxis, ylims=(-0.1, 1.1), ylabel=yaxis_constraint_label, kwargs...)

        for ci in 1:lastindex(constraints)
            isnothing(constraints[ci]) && continue
            c_vals = constraints[ci].(xs)
            label = isnothing(constraint_labels) ? "constraint_$ci" : constraint_labels[ci]
            if isnothing(constraint_colors)
                plot!(constraint_plot, reduce(vcat, xs), reduce(vcat, c_vals); label)
            else
                plot!(constraint_plot, reduce(vcat, xs), reduce(vcat, c_vals); label, color=constraint_colors[ci])
            end
        end
    end


    # DATA PLOT
    data_plot = Plots.plot(; xaxis, yaxis, title, ylabel=yaxis_data_label, kwargs...)

    # obj func and data
    label = isnothing(obj_func_label) ? "obj_func" : obj_func_label
    plot!(data_plot, x->obj_func([x]), domain_lb[1], domain_ub[1]; label, color=(isnothing(obj_color) ? :green : obj_color))
    if init_data > 0
        scatter!(data_plot, x_data[1:init_data], y_data[1:init_data]; label="initial data", color=:yellow)
        scatter!(data_plot, x_data[init_data+1:end], y_data[init_data+1:end]; label="requested data", color=:brown)
    else
        scatter!(data_plot, x_data, y_data; label="data", color=:orange)
    end

    # models
    for i in 1:lastindex(models)
        isnothing(models[i][1]) && continue
        x_range = (xaxis == :log) ? log_range(domain_lb[1],domain_ub[1],points) : collect(LinRange([domain_lb[1]],[domain_ub[1]],points))
        y_range = reduce(vcat, models[i][1].(x_range))
        var_range = isnothing(models[i][2]) ? nothing : getindex.(models[i][2].(x_range), 1)
        label = isnothing(model_labels) ? (length(models) > 1 ? "model_$i" : "model") : model_labels[i]
        color = isnothing(model_colors) ? :red : model_colors[i]
        plot!(data_plot, reduce(vcat, x_range), reduce(vcat, y_range); label, ribbon=var_range, points, color)
    end
    
    # COMBINED PLOT
    plots = [data_plot]
    isnothing(constraint_plot) || push!(plots, constraint_plot)
    isnothing(util_plot) || push!(plots, util_plot)
    p = Plots.plot(plots...; layout=(length(plots),1), legend=:outerright, minorgrid=true, xlabel=xaxis_label, kwargs...)
    show_plot && display(p)
    return p
end

function model_dim_slice(model, dim)
    μ = isnothing(model[1]) ? nothing : x -> model[1](x)[dim]
    σ = isnothing(model[2]) ? nothing : x -> model[2](x)[dim]
    return μ, σ
end

function plot_res_2x1(model, obj_func; points=200)
    x_range = range(A[1], B[1]; length=points)
    y_range = range(A[2], B[2]; length=points)

    surface(x_range, y_range, (x,y)->obj_func([x,y])[1]; color=:blues, alpha=0.4)
    surface!(x_range, y_range, (x,y)->model([x,y])[1]; color=:reds, alpha=0.4)
end

function plot_bsf_boxplots(results; show_plot=true, labels=nothing)
    p = plot(; title="Best-so-far solutions found\n(medians with 1st and 3rd quartiles)", legend=:topleft)

    for i in 1:lastindex(results)
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