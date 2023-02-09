using Plots; pyplot()
using LaTeXStrings
using Statistics
using Optim

# UNUSED

# TODO: refactor needed

function create_plots(f_true, utils, util_opts, models, model_samples, constraints, X, Y;
    iter,
    domain,
    init_data_size,
    show_plots,
    param_fit_alg,
    samples_lable,
    kwargs...
)
    y_dim = size(Y)[1]
    x_dim = size(X)[1]
    (x_dim != 1) && throw(ArgumentError("Input dimension must be equal to 1!"))

    colors = [:red, :purple, :blue]
    model_labels = ["param", "semiparam", "nonparam"]
    u_label = isnothing(constraints) ? "EI" : "cwEI"
    util_labels = ["param", "semiparam", "nonparam"]
    model_sample_labels = isnothing(model_samples) ? nothing : [samples_lable, [nothing for _ in 1:length(model_samples)]...]
    model_sample_colors = isnothing(model_samples) ? nothing : [:black for _ in 1:length(model_samples)]

    sliced_models = [model_dim_slice.(models, Ref(d)) for d in 1:y_dim]
    sliced_obj = [x -> f_true(x)[d] for d in 1:y_dim]
    p = plot_res_1x1(sliced_models, sliced_obj, X, Y, domain, constraints;
        utils,
        util_opts,
        title="ITER $iter",
        xaxis_label="x",
        init_data=init_data_size,
        model_colors=colors,
        util_colors=colors,
        model_labels,
        util_labels,
        show_plot=show_plots,
        yaxis_util_label=u_label,
        model_samples,
        model_sample_labels,
        model_sample_colors,
        kwargs...
    )
    return p
end

function plot_res_1x1(sliced_models, sliced_obj, x_data, y_data, domain, constraints;
    model_samples=nothing,
    utils=nothing,
    util_opts=nothing,
    points=200,
    title="",
    show_plot=true,
    xaxis=:identity,
    yaxis=:identity,
    obj_func_label=nothing,
    model_labels=nothing,
    model_sample_labels=nothing,
    util_labels=nothing,
    xaxis_label=nothing,
    yaxis_util_label="normalized utility",
    init_data=0,
    obj_color=nothing,
    model_colors=nothing,
    model_sample_colors=nothing,
    util_colors=nothing,
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

    # DATA PLOT
    data_plots = Plots.Plot[]
    for d in eachindex(sliced_obj)
        dp = Plots.plot(; xaxis, yaxis, ylabel="y$d", kwargs...)

        # model samples
        if !isnothing(model_samples)
            for i in 1:lastindex(model_samples)
                x_range = (xaxis == :log) ? log_range(domain_lb[1],domain_ub[1],points) : collect(LinRange([domain_lb[1]],[domain_ub[1]],points))
                y_range = reduce(vcat, (x->model_samples[i](x)[1][d]).(x_range))
                var_range = nothing
                label = isnothing(model_sample_labels) ? "sample_$i" : model_sample_labels[i]
                color = isnothing(model_sample_colors) ? :red : model_sample_colors[i]
                plot!(dp, reduce(vcat, x_range), reduce(vcat, y_range); label, linestyle=:dash, linealpha=0.2, ribbon=var_range, points, color)
            end
        end

        # models
        models = sliced_models[d]
        for i in 1:lastindex(models)
            isnothing(models[i]) && continue
            x_range = (xaxis == :log) ? log_range(domain_lb[1],domain_ub[1],points) : collect(LinRange(domain_lb[1],domain_ub[1],points))
            y_range = (x->models[i]([x])[1]).(x_range)
            var_range = (x->models[i]([x])[2]).(x_range)
            label = isnothing(model_labels) ? (length(models) > 1 ? "model_$i" : "model") : model_labels[i]
            color = isnothing(model_colors) ? :red : model_colors[i]
            plot!(dp, x_range, y_range; label, ribbon=var_range, points, color)
        end

        # obj func and data
        label = isnothing(obj_func_label) ? "obj_func" : obj_func_label
        if (!isnothing(constraints) && !isinf(constraints[d]))
            plot!(dp, x->constraints[d], domain_lb[1], domain_ub[1]; label="constraint", color=:black, linestyle=:dot, thickness_scaling=2)
        end
        plot!(dp, x->sliced_obj[d]([x]), domain_lb[1], domain_ub[1]; label, color=(isnothing(obj_color) ? :green : obj_color))
        if init_data > 0
            scatter!(dp, x_data[1:init_data], y_data[d,1:init_data]; label="initial data", color=:yellow)
            scatter!(dp, x_data[init_data+1:end], y_data[d,init_data+1:end]; label="requested data", color=:brown)
        else
            scatter!(dp, x_data, y_data[d,:]; label="data", color=:orange)
        end

        push!(data_plots, dp)
    end
    
    # COMBINED PLOT
    plots = data_plots
    isnothing(util_plot) || push!(plots, util_plot)
    plot!(first(plots); title)
    plot!(last(plots); xlabel=xaxis_label)

    p = Plots.plot(plots...; layout=(length(plots),1), legend=:outerright, minorgrid=true, kwargs...)
    show_plot && display(p)
    return p
end

function model_dim_slice(model, dim)
    function model_slice(x)
        mean, var = model(x)
        mean[dim], var[dim]
    end
end
model_dim_slice(model::Nothing, dim) = nothing

function plot_res_2x1(model, obj_func; points=200)
    x_range = range(A[1], B[1]; length=points)
    y_range = range(A[2], B[2]; length=points)

    surface(x_range, y_range, (x,y)->obj_func([x,y])[1]; color=:blues, alpha=0.4)
    surface!(x_range, y_range, (x,y)->model([x,y])[1]; color=:reds, alpha=0.4)
end

function plot_bsf_boxplots(results; show_plot=true, labels=["param", "semiparam", "nonparam"])
    p = plot(; title="Best-so-far solutions found\n(min,median,max)", legend=:topleft)

    for i in 1:lastindex(results)
        bsf_data = getfield.(results[i], :bsf)
        bsf_data = reduce(hcat, bsf_data)
        for i in eachindex(bsf_data) if isnothing(bsf_data[i]) bsf_data[i] = 0. end end
        label = isnothing(labels) ? nothing : labels[i]
        y_vals = median.(eachrow(bsf_data))
        y_err_l = y_vals .- quantile.(eachrow(bsf_data), 0.)  # 0.25
        y_err_u = quantile.(eachrow(bsf_data), 1.) .- y_vals  # 0.75
        plot!(p, y_vals; yerror=(y_err_l, y_err_u), label, markerstrokecolor=:auto, legend=:best)
    end

    show_plot && display(p)
    return p
end

function plot_paramdiff_boxplots(true_val, param_idx, results; show_plot=true, labels=["param", "semiparam", "nonparam"])
    p = plot(; title="Param-diff\n(min,median,max)", legend=:topleft)

    for i in 1:2  # exclude nonparam
        data = getfield.(results[i], :parameters)
        data = reduce(hcat, data)
        data = getindex.(data, param_idx)
        data = abs.(data .- true_val)
        label = isnothing(labels) ? nothing : labels[i]
        y_vals = median.(eachrow(data))
        y_err_l = y_vals .- quantile.(eachrow(data), 0.)  # 0.25
        y_err_u = quantile.(eachrow(data), 1.) .- y_vals  # 0.75
        plot!(p, y_vals; yerror=(y_err_l, y_err_u), label, markerstrokecolor=:auto, legend=:best)
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
