using Plots
using LinearAlgebra
using Optim
using Distributions
using Soss
using SampleChainsDynamicHMC
using Stheno

include("model.jl")
include("plotting.jl")
include("utils.jl")

# Sample from the posterior parameter distributions given the data 'X', 'Y'.
function sample_posterior(model, X, Y; sample_count=4000)
    q = zeros(length(model.params) + 1)  # workaround: https://discourse.julialang.org/t/dynamichmc-reached-maximum-number-of-iterations/24721
    return Soss.sample(model.prob_model(X=collect(eachrow(X))) | (Y=Y,), dynamichmc(; init=(q=q,)), sample_count)
end

# Bayesian optimization with a Nx1 dimensional parametric model.
# Assumes Gaussian noise.
# Plotting only works for 1D->1D problems.
function boss(f, X, Y, model, domain_lb, domain_ub;
    sample_count=200,
    util_opt_multistart=100,
    info=true,
    make_plots=false,
    show_plots=true,
    plot_all_models=false,
    max_iters=nothing,
    test_X=nothing,
    test_Y=nothing,
    target_err=nothing,
    kernel=Matern52Kernel(),
    use_model=:semiparam,  # :param, :semiparam, :nonparam
    kwargs...
)
    init_data_size = size(X)[1]

    plots = make_plots ? Plots.Plot[] : nothing
    bsf = [maximum(skipnothing(Y))]
    errs = (isnothing(test_X) || isnothing(test_Y)) ? nothing : Float64[]

    i = 0
    opt_x_ = 0.
    while true
        info && print("\nITER $i\n")

        # NEW DATA - - - - - - - -
        if i != 0
            info && print("  evaluating the objective function ...\n")
            x_ = opt_x_
            y_ = f(x_)

            info && print("  new data-point: ($x_, $y_)\n")
            X = vcat(X, x_')
            Y = vcat(Y, y_)

            if y_ > last(bsf)
                push!(bsf, y_)
            else
                push!(bsf, last(bsf))
            end
        end

        # MODEL INFERENCE - - - - - - - -
        info && print("  model inference ...\n")

        # parametric model
        if plot_all_models || (use_model == :param) || (use_model == :semiparam)
            post = sample_posterior(model, X, Y; sample_count)
            param_samples = hcat(getproperty.(Ref(post), model.params)...)
            ϵ_samples = rand(Distributions.Normal(), sample_count)  # TODO Refactor? (This assumes Gaussian noise.)
            noise = mean(post.σ)
            parametric = (x -> model_predict_MC(model.predict, x; param_samples, ϵ_samples, noise, sample_count), nothing)
        else
            parametric = (nothing, nothing)
        end

        # semiparametric model (param + GP)
        if plot_all_models || (use_model == :semiparam)
            semi_gp = GP(parametric[1], kernel)
            semi_gp_ = posterior(semi_gp(X'), Y)
            semiparametric = (x -> mean(semi_gp_([x]))[1], x -> var(semi_gp_([x]))[1])
        else
            semiparametric = (nothing, nothing)
        end

        # nonparametric model (GP)
        if plot_all_models || (use_model == :nonparam)
            pure_gp = GP(kernel)
            pure_gp_ = posterior(pure_gp(X'), Y)
            nonparametric = (x -> mean(pure_gp_([x]))[1], x -> var(pure_gp_([x]))[1])
        else
            nonparametric = (nothing, nothing)
        end

        # UTILITY MAXIMIZATION - - - - - - - -
        info && print("  optimizing utility ...\n")
        multistart = util_opt_multistart
        res_param = res_semiparam = res_nonparam = nothing
        acq_param = acq_semiparam = acq_nonparam = nothing
        
        # parametric
        if plot_all_models || (use_model == :param)
            acq_param(x) = EI_MC(model.predict, x; param_samples, ϵ_samples, noise, best_yet=last(bsf), sample_count)
            res_param = opt_acquisition(acq_param, domain_lb, domain_ub; multistart, info)
        end

        # semiparametric
        if plot_all_models || (use_model == :semiparam)
            acq_semiparam(x) = EI_normal(semi_gp_, x; best_yet=last(bsf))
            res_semiparam = opt_acquisition(acq_semiparam, domain_lb, domain_ub; multistart, info)
        end

        # nonparametric
        if plot_all_models || (use_model == :nonparam)
            acq_nonparam(x) = EI_normal(pure_gp_, x; best_yet=last(bsf))
            res_nonparam = opt_acquisition(acq_nonparam, domain_lb, domain_ub; multistart, info)
        end

        opt_x_, err = select_opt_x_and_calculate_error(use_model, res_param, res_semiparam, res_nonparam, parametric, semiparametric, nonparametric, !isnothing(errs), test_X, test_Y)
        isnothing(errs) || push!(errs, err)
        info && print("  optimal next x: $opt_x_\n")
        info && print("  model error: $err\n")

        # PLOTTING - - - - - - - -
        if make_plots
            info && print("  plotting ...\n")
            colors = [:red, :purple, :blue]
            labels = ["param", "semiparam", "nonparam"]
            models = [parametric, semiparametric, nonparametric]
            utils = [acq_param, acq_semiparam, acq_nonparam]
            util_opts = [res_param, res_semiparam, res_nonparam]
            p = plot_res_1x1(models, f, X, Y, domain_lb, domain_ub; utils, util_opts, title="ITER $i", init_data=init_data_size, model_colors=colors, util_colors=colors, model_labels=labels, util_labels=labels.*" EI", show_plot=show_plots, kwargs...)
            push!(plots, p)
        end

        # model error calculation is moved to 'select_opt_x_and_calculate_error' for now

        # # CALCULATE MODEL ERROR - - - - - - - -
        # if !(isnothing(errs))
        #     param_err = rms_error(test_X, test_Y, parametric[1])
        #     semiparam_err = rms_error(test_X, test_Y, semiparametric[1])
        #     nonparam_err = rms_error(test_X, test_Y, nonparametric[1])
        #     INFO && print("  model errors: $((param_err, semiparam_err, nonparam_err))\n")
        #     push!(errs[1], param_err)
        #     push!(errs[2], semiparam_err)
        #     push!(errs[3], nonparam_err)
        # end

        # TERMINATION CONDITIONS - - - - - - - -
        if !isnothing(max_iters)
            (i >= max_iters) && break
        end
        if !isnothing(target_err)
            (err < target_err) && break
        end
        i += 1
    end

    return X, Y, bsf, errs, plots
end

function select_opt_x_and_calculate_error(use_model, res_param, res_semiparam, res_nonparam, parametric, semiparametric, nonparametric, calc_err, test_X, test_Y)
    opt_results = (res_param, res_semiparam, res_nonparam)
    models = (parametric, semiparametric, nonparametric)
    
    if use_model == :param
        m = 1
    elseif use_model == :semiparam
        m = 2
    elseif use_model == :nonparam
        m = 3
    end
    
    opt_x_ = opt_results[m][1]
    err = calc_err ? rms_error(test_X, test_Y, models[m][1]) : nothing
    return opt_x_, err
end

function model_predict_MC(model_predict, x; param_samples, ϵ_samples, noise, sample_count)
    val = 0.
    for i in 1:sample_count
        val += model_predict(x, param_samples[i,:]...) # + (noise * ϵ_samples[i])  # The noise sum should approach zero so it is unnecessary.
    end
    return val / sample_count
end

function opt_acquisition(acq, domain_lb, domain_ub; multistart=1, info=true)
    dim = length(domain_lb)
    starts = rand(dim, multistart) .* (domain_ub .- domain_lb) .+ domain_lb
    best_res = (0., -Inf)
    for i in 1:multistart
        s = starts[:,i]
        try
            opt_res = Optim.optimize(x -> -acq(x), domain_lb, domain_ub, s, Fminbox(LBFGS()))
            res = Optim.minimizer(opt_res), -Optim.minimum(opt_res)
            (res[2] > best_res[2]) && (best_res = res) 
        catch
            info && print("    Optimization failed to converge!\n")
        end
    end
    return best_res
end

function EI_MC(model_predict, x; param_samples, ϵ_samples, noise, best_yet, sample_count)
    val = 0.
    for i in 1:sample_count
        val += max(0, model_predict(x, param_samples[i,:]...) + (noise * ϵ_samples[i]) - best_yet)
    end
    return val / sample_count
end

function EI_normal(gp, x; best_yet)
    μy = mean(gp([x]))[1]
    σy = var(gp([x]))[1]
    norm_ϵ = (μy - best_yet) / σy
    return (μy - best_yet) * cdf(Distributions.Normal(), norm_ϵ) + σy * pdf(Distributions.Normal(), norm_ϵ)
end
