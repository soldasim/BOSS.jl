using Plots; pyplot()
using LaTeXStrings
using Statistics
using Optim

function create_plots(f_true, utils, util_opts, models, model_samples, feas_probs, X, Y;
    iter,
    y_dim,
    feasibility,
    feasibility_count,
    domain,
    init_data_size,
    show_plots,
    param_fit_alg,
    samples_lable,
    kwargs...
)
    colors = [:red, :purple, :blue]
    model_labels = (param_fit_alg == :LBFGS) ?
        ["param\n(MLE fit)", "semiparam\n(MLE fit)", "nonparam\n(MLE fit)"] :
        ["param\n(best sample)", "semiparam\n(best sample)", "nonparam\n(best sample)"]
    u_label = feasibility ? "cwEI" : "EI"
    util_labels = (param_fit_alg == :LBFGS) ?
        ["param\n(LBFGS)", "semiparam\n(LBFGS)", "nonparam\n(LBFGS)"] :
        ["param\n(NUTS)", "semiparam\n(NUTS)", "nonparam\n(NUTS)"]
    feasibility_funcs = feasibility ? [x->feas_probs(x)[i] for i in 1:feasibility_count] : nothing
    model_sample_labels = isnothing(model_samples) ? nothing : [samples_lable, [nothing for _ in 1:length(model_samples)]...]
    model_sample_colors = isnothing(model_samples) ? nothing : [:black for _ in 1:length(model_samples)]

    plots = Plots.Plot[]
    for d in 1:y_dim
        title = (y_dim > 1) ? "ITER $iter, DIM $d" : "ITER $iter"
        models = model_dim_slice.(models, d)
        
        p = plot_res_1x1(models, x -> f_true(x)[d], X, Y, domain;
            utils,
            util_opts,
            feasibility_funcs,
            yaxis_feasibility_label="feasibility constraint\nsatisfaction prob.",
            title,
            init_data=init_data_size,
            model_colors=colors,
            util_colors=colors,
            model_labels,
            util_labels,
            show_plot=show_plots,
            yaxis_util_label=u_label,
            model_samples=model_samples,
            model_sample_labels,
            model_sample_colors,
            kwargs...
        )
        push!(plots, p)
    end
    return plots
end

function plot_res_1x1(models, obj_func, x_data, y_data, domain;
    model_samples=nothing,
    utils=nothing,
    util_opts=nothing,
    feasibility_funcs=nothing,
    points=200,
    title="",
    show_plot=true,
    xaxis=:identity,
    yaxis=:identity,
    obj_func_label=nothing,
    model_labels=nothing,
    model_sample_labels=nothing,
    util_labels=nothing,
    feasibility_labels=nothing,
    xaxis_label=nothing,
    yaxis_util_label="normalized utility",
    yaxis_data_label=nothing,
    yaxis_feasibility_label="feasibility",
    init_data=0,
    obj_color=nothing,
    model_colors=nothing,
    model_sample_colors=nothing,
    util_colors=nothing,
    feasibility_colors=nothing,
    kwargs...
)
    bounds = get_bounds(domain)
    domain_lb, domain_ub = bounds[1], bounds[2]
    xs = LinRange(domain_lb, domain_ub, points)

    # UTIL PLOT
    util_plot = nothing

    if !isnothing(utils)
        util_plot = Plots.plot(; xaxis, yaxis=(:log, [0.09, 1.1]), ylabel=yaxis_util_label, kwargs...)

        for ui in 1:lastindex(utils)
            isnothing(utils[ui]) && continue
            u_vals = util_vals(utils[ui], xs, domain)
            if !(isnothing(util_opts) || isnothing(util_opts[ui]))
                u_opt_x, u_opt_y = util_opts[ui]
                u_max = max(u_opt_y, maximum(u_vals))
                if u_max > 0.
                    u_vals = u_vals .* ((1/u_max) * (9/10)) .+ 0.1
                    u_opt_y = u_opt_y .* ((1/u_max) * (9/10)) .+ 0.1
                end
                scatter!(util_plot, u_opt_x, [u_opt_y]; label="", color=util_colors[ui])
            end
            label = isnothing(util_labels) ? "utility_$ui" : util_labels[ui]
            if isnothing(util_colors)
                plot!(util_plot, reduce(vcat, xs), reduce(vcat, u_vals); label)
            else
                plot!(util_plot, reduce(vcat, xs), reduce(vcat, u_vals); label, color=util_colors[ui])
            end
        end
    end

    # FEASIBILITY PLOT
    feasibility_plot = nothing

    if !isnothing(feasibility_funcs)
        feasibility_plot = Plots.plot(; xaxis, ylims=(-0.1, 1.1), ylabel=yaxis_feasibility_label, kwargs...)

        for fi in 1:lastindex(feasibility_funcs)
            isnothing(feasibility_funcs[fi]) && continue
            f_vals = feasibility_funcs[fi].(xs)
            label = isnothing(feasibility_labels) ? "feasibility_$fi" : feasibility_labels[fi]
            if isnothing(feasibility_colors)
                plot!(feasibility_plot, reduce(vcat, xs), reduce(vcat, f_vals); label)
            else
                plot!(feasibility_plot, reduce(vcat, xs), reduce(vcat, f_vals); label, color=feasibility_colors[fi])
            end
        end
    end


    # DATA PLOT
    data_plot = Plots.plot(; xaxis, yaxis, title, ylabel=yaxis_data_label, kwargs...)

    # model samples
    if !isnothing(model_samples)
        for i in 1:lastindex(model_samples)
            x_range = (xaxis == :log) ? log_range(domain_lb[1],domain_ub[1],points) : collect(LinRange([domain_lb[1]],[domain_ub[1]],points))
            y_range = reduce(vcat, model_samples[i][1].(x_range))
            var_range = nothing  # isnothing(model_samples[i][2]) ? nothing : getindex.(model_samples[i][2].(x_range), 1)
            label = isnothing(model_sample_labels) ? "sample_$i" : model_sample_labels[i]
            color = isnothing(model_sample_colors) ? :red : model_sample_colors[i]
            plot!(data_plot, reduce(vcat, x_range), reduce(vcat, y_range); label, linestyle=:dash, linealpha=0.2, ribbon=var_range, points, color)
        end
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

    # obj func and data
    label = isnothing(obj_func_label) ? "obj_func" : obj_func_label
    plot!(data_plot, x->obj_func([x]), domain_lb[1], domain_ub[1]; label, color=(isnothing(obj_color) ? :green : obj_color))
    if init_data > 0
        scatter!(data_plot, x_data[1:init_data], y_data[1:init_data]; label="initial data", color=:yellow)
        scatter!(data_plot, x_data[init_data+1:end], y_data[init_data+1:end]; label="requested data", color=:brown)
    else
        scatter!(data_plot, x_data, y_data; label="data", color=:orange)
    end
    
    # COMBINED PLOT
    plots = [data_plot]
    isnothing(feasibility_plot) || push!(plots, feasibility_plot)
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

function plot_bsf_boxplots(results; show_plot=true, labels=["param", "semiparam", "nonparam"])
    p = plot(; title="Best-so-far solutions found\n(medians with 1st and 3rd quartiles)", legend=:topleft)

    for i in 1:lastindex(results)
        bsf_data = getfield.(results[i], :bsf)
        bsf_data = reduce(hcat, bsf_data)
        for i in eachindex(bsf_data) if isnothing(bsf_data[i]) bsf_data[i] = 0. end end
        label = isnothing(labels) ? nothing : labels[i]
        y_vals = median.(eachrow(bsf_data))
        y_err_l = y_vals .- quantile.(eachrow(bsf_data), 0.25)
        y_err_u = quantile.(eachrow(bsf_data), 0.75) .- y_vals
        plot!(p, y_vals; yerror=(y_err_l, y_err_u), label, markerstrokecolor=:auto)
    end

    show_plot && display(p)
    return p
end

function plot_model_sample(model, domain; xaxis=:identity, label=nothing, color=:red, points=200, show_plot=true)
    x_range = (xaxis == :log) ? log_range(domain[1][1], domain[2][1], points) : collect(LinRange([domain[1][1]], [domain[2][1]], points))
    y_range = reduce(vcat, model[1].(x_range))
    var_range = isnothing(model[2]) ? nothing : getindex.(model[2].(x_range), 1)
    p = plot(reduce(vcat, x_range), reduce(vcat, y_range); label, ribbon=var_range, points, color)
    show_plot && display(p)
    return p
end

function util_vals(util, xs, domain::Tuple)
    return util.(xs)
end
function util_vals(util, xs, domain)
    interior = Boss.in_domain.(xs, Ref(domain))
    return [(interior[i] ? util(xs[i]) : 0.) for i in eachindex(xs)]
end
