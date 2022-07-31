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
#
# Uncostrained -> boss(f, X, Y, model, domain_lb, domain_ub; kwargs...)
#              ->   f(x) = y
#
# Constrained  -> boss(f, X, Y, Z, model, domain_lb, domain_ub; kwargs...)
#              ->   f(x) = (y, g1, g2, ...)
function boss(f, X, Y, model, domain_lb, domain_ub; kwargs...)
    return boss(f, X, Y, nothing, model, domain_lb, domain_ub; kwargs...)
end

function boss(f, X, Y, Z, model, domain_lb, domain_ub;
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
    c_kernel=kernel,
    use_model=:semiparam,  # :param, :semiparam, :nonparam
    kwargs...
)
    constrained = !isnothing(Z)
    c_count = constrained ? size(Z)[2] : 0
    init_data_size = size(X)[1]

    plots = make_plots ? Plots.Plot[] : nothing
    bsf = [maximum(skipnothing(Y))]
    errs = (isnothing(test_X) || isnothing(test_Y)) ? nothing : Float64[]

    i = 0
    opt_new_x = 0.
    while true
        info && print("\nITER $i\n")

        # NEW DATA - - - - - - - -
        if i != 0
            info && print("  evaluating the objective function ...\n")
            x_ = opt_new_x
            y_, z_... = f(x_)
            z_ = [z_...]

            info && print("  new data-point: ($x_, $y_)\n")
            X = vcat(X, x_')
            Y = vcat(Y, y_)
            
            if constrained
                Z = vcat(Z, z_')
                feasible = all(z_ .>= 0)
                info && (feasible ? print("    feasible\n") : print("    infeasible\n"))
            else
                feasible = true
            end

            if feasible && (y_ > last(bsf))
                push!(bsf, y_)
            else
                push!(bsf, last(bsf))
            end
        end

        # MODEL INFERENCE - - - - - - - -
        info && print("  model inference ...\n")

        # parametric model
        if plot_all_models || (use_model == :param) || (use_model == :semiparam)
            post_ = sample_posterior(model, X, Y; sample_count)
            param_samples = hcat(getproperty.(Ref(post_), model.params)...)
            ϵ_samples = rand(Distributions.Normal(), sample_count)  # TODO Refactor? (This assumes Gaussian noise.)
            noise = mean(post_.σ)
            parametric = (x -> model_predict_MC(model.predict, x; param_samples, ϵ_samples, noise, sample_count), nothing)
        else
            parametric = (nothing, nothing)
        end

        # semiparametric model (param + GP)
        if plot_all_models || (use_model == :semiparam)
            semi_gp_ = GP(parametric[1], kernel)
            semi_gp_ = posterior(semi_gp_(X'), Y)
            semiparametric = get_pred_distr(semi_gp_)
        else
            semiparametric = (nothing, nothing)
        end

        # nonparametric model (GP)
        if plot_all_models || (use_model == :nonparam)
            pure_gp_ = GP(kernel)
            pure_gp_ = posterior(pure_gp_(X'), Y)
            nonparametric = get_pred_distr(pure_gp_)
        else
            nonparametric = (nothing, nothing)
        end

        # constraint models (GPs)
        c_gps_ = [posterior(GP(c_kernel)(X'), Z[:,i]) for i in 1:c_count]
        c_models = get_pred_distr.(c_gps_)

        # UTILITY MAXIMIZATION - - - - - - - -
        info && print("  optimizing utility ...\n")
        multistart = util_opt_multistart
        res_param = res_semiparam = res_nonparam = nothing
        acq_param = acq_semiparam = acq_nonparam = nothing
        
        # parametric
        if plot_all_models || (use_model == :param)
            ei_param_(x) = EI_MC(model.predict, x; param_samples, ϵ_samples, noise, best_yet=last(bsf), sample_count)
            acq_param(x) = constraint_weighted_acq(ei_param_(x), x, c_models)
            res_param = opt_acquisition(acq_param, domain_lb, domain_ub; multistart, info)
        end

        # semiparametric
        if plot_all_models || (use_model == :semiparam)
            ei_semiparam_(x) = EI_GP(semiparametric, x; best_yet=last(bsf))
            acq_semiparam(x) = constraint_weighted_acq(ei_semiparam_(x), x, c_models)
            res_semiparam = opt_acquisition(acq_semiparam, domain_lb, domain_ub; multistart, info)
        end

        # nonparametric
        if plot_all_models || (use_model == :nonparam)
            ei_nonparam_(x) = EI_GP(nonparametric, x; best_yet=last(bsf))
            acq_nonparam(x) = constraint_weighted_acq(ei_nonparam_(x), x, c_models)
            res_nonparam = opt_acquisition(acq_nonparam, domain_lb, domain_ub; multistart, info)
        end

        opt_new_x, err = select_opt_x_and_calculate_error(use_model, res_param, res_semiparam, res_nonparam, parametric, semiparametric, nonparametric, !isnothing(errs), test_X, test_Y)
        isnothing(errs) || push!(errs, err)
        info && print("  optimal next x: $opt_new_x\n")
        info && print("  model error: $err\n")

        # PLOTTING - - - - - - - -
        if make_plots
            info && print("  plotting ...\n")
            colors = [:red, :purple, :blue]
            labels = ["param", "semiparam", "nonparam"]
            models = [parametric, semiparametric, nonparametric]
            utils = [acq_param, acq_semiparam, acq_nonparam]
            util_opts = [res_param, res_semiparam, res_nonparam]
            util_label = constrained ? "cwEI" : "EI"
            constraints = [constraint_prob(c) for c in c_models]
            p = plot_res_1x1(models, f, X, Y, domain_lb, domain_ub; utils, util_opts, constraints, yaxis_constraint_label="constraint\nsatisfaction probability", title="ITER $i", init_data=init_data_size, model_colors=colors, util_colors=colors, model_labels=labels, util_labels=labels, show_plot=show_plots, yaxis_util_label=util_label, kwargs...)
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

    return X, Y, Z, bsf, errs, plots
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
    
    opt_new_x = opt_results[m][1]
    err = calc_err ? rms_error(test_X, test_Y, models[m][1]) : nothing
    return opt_new_x, err
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
    convergence_errors = 0
    for i in 1:multistart
        s = starts[:,i]
        try
            opt_res = Optim.optimize(x -> -acq(x), domain_lb, domain_ub, s, Fminbox(LBFGS()))
            res = Optim.minimizer(opt_res), -Optim.minimum(opt_res)
            (res[2] > best_res[2]) && (best_res = res) 
        catch
            convergence_errors += 1
        end
    end
    info && (convergence_errors > 0) && print("    $convergence_errors/$multistart optimization runs failed to converge!\n")
    return best_res
end

function get_pred_distr(gp)
    μy(x) = mean(gp([x]))[1]
    σy(x) = var(gp([x]))[1]
    return μy, σy
end
function get_pred_distr(gp, x)
    μy, σy = get_pred_distr(gp)
    return μy(x), σy(x)
end

function EI_MC(model_predict, x; param_samples, ϵ_samples, noise, best_yet, sample_count)
    val = 0.
    for i in 1:sample_count
        val += max(0, model_predict(x, param_samples[i,:]...) + (noise * ϵ_samples[i]) - best_yet)
    end
    return val / sample_count
end

function EI_GP(model, x; best_yet)
    μy, σy = model[1](x), model[2](x)
    norm_ϵ = (μy - best_yet) / σy
    return (μy - best_yet) * cdf(Distributions.Normal(), norm_ϵ) + σy * pdf(Distributions.Normal(), norm_ϵ)
end

function constraint_weighted_acq(acq, x, constraint_models)
    for c in constraint_models
        acq *= constraint_prob(c)(x)
    end
    return acq
end

function constraint_prob(c_model)
    function p(x)
        μ, σ = c_model[1](x), c_model[2](x)
        distr = Distributions.Normal(μ, σ)
        return (1. - cdf(distr, 0))
    end
    return p
end
