using Plots
using LinearAlgebra
using Optim
using Distributions
using SampleChainsDynamicHMC
using Stheno

include("model.jl")
include("plotting.jl")
include("utils.jl")

# Sample from the posterior parameter distributions given the data 'X', 'Y'.
function sample_posterior(model_prob, X, Y; sample_count=4000)
    return Soss.sample(model_prob(X=collect(eachrow(X))) | (Y=Y,), dynamichmc(), sample_count)
end

# Bayesian optimization with a Nx1 dimensional parametric model.
function optim!(f, X, Y, model, domain_lb, domain_ub;
    sample_count=200,
    util_opt_multistart=200,
    INFO=true,
    max_iters=nothing,
    X_test=nothing,
    Y_test=nothing,
    target_err=nothing,
    kwargs...
)
    MODEL_COUNT = 3
    init_data_size = size(X)[1]

    plots = Plots.Plot[]
    errs = (isnothing(X_test) || isnothing(Y_test)) ? nothing : [Float64[] for _ in 1:MODEL_COUNT]

    i = 0
    opt_x_ = 0.
    while true
        INFO && print("\nITER $i\n")

        # NEW DATA - - - - - - - -
        if i != 0
            INFO && print("  evaluating the objective function ...\n")
            x_ = opt_x_
            y_ = f(x_)

            INFO && print("  new data-point: ($x_, $y_)\n")
            X = vcat(X, x_)
            Y = vcat(Y, y_)
        end

        # MODEL INFERENCE - - - - - - - -
        INFO && print("  model inference ...\n")

        # parametric model
        post = sample_posterior(model.prob_model, X, Y; sample_count)
        param_samples = hcat(getproperty.(Ref(post), model.params)...)
        ϵ_samples = rand(Distributions.Normal(), sample_count)
        noise = mean(post.σ)
        parametric(x) = model_predict_MC(model.predict, x; param_samples, ϵ_samples, noise, sample_count)
        
        # semiparametric model (param + GP)
        semi_gp = GP(parametric, Matern52Kernel())
        semi_gp_ = posterior(semi_gp(X'), Y)
        semiparametric(x) = mean(semi_gp_([x]))[1]

        # nonparametric model (GP)
        pure_gp = GP(Matern52Kernel())
        pure_gp_ = posterior(pure_gp(X'), Y)
        nonparametric(x) = mean(pure_gp_([x]))[1]

        # UTILITY MAXIMIZATION - - - - - - - -
        INFO && print("  optimizing utility ...\n")
        multistart = util_opt_multistart
        
        # parametric
        acq_param(x) = EI_MC(model.predict, x; param_samples, ϵ_samples, noise, best_yet=maximum(Y), sample_count)
        res_param = opt_acquisition(acq_param, domain_lb, domain_ub; multistart)
        
        # semiparametric
        acq_semiparam(x) = EI_normal(semi_gp_, x; best_yet=maximum(Y))
        res_semiparam = opt_acquisition(acq_semiparam, domain_lb, domain_ub; multistart)
        
        # nonparametric
        acq_nonparam(x) = EI_normal(pure_gp_, x; best_yet=maximum(Y))
        res_nonparam = opt_acquisition(acq_nonparam, domain_lb, domain_ub; multistart)

        opt_x_ = res_semiparam[1]
        INFO && print("  optimal next x: $opt_x_\n")

        # PLOTTING - - - - - - - -
        INFO && print("  plotting ...\n")
        colors = [:red, :purple, :blue]
        labels = ["param", "semiparam", "nonparam"]
        models = [parametric, semiparametric, nonparametric]
        utils = [acq_param, acq_semiparam, acq_nonparam]
        util_opts = [res_param, res_semiparam, res_nonparam]
        p = plot_res_1x1(models, f, X, Y, domain_lb, domain_ub; utils, util_opts, title="ITER $i", init_data=init_data_size, model_colors=colors, util_colors=colors, model_labels=labels, util_labels=labels.*" EI", kwargs...)
        push!(plots, p)

        # CALCULATE MODEL ERROR - - - - - - - -
        if !(isnothing(X_test) || isnothing(Y_test))
            param_err = rms_error(X_test, Y_test, parametric)
            semiparam_err = rms_error(X_test, Y_test, semiparametric)
            nonparam_err = rms_error(X_test, Y_test, nonparametric)
            INFO && print("  model errors: $((param_err, semiparam_err, nonparam_err))\n")
            push!(errs[1], param_err)
            push!(errs[2], semiparam_err)
            push!(errs[3], nonparam_err)
        end

        # TERMINATION CONDITIONS - - - - - - - -
        if !isnothing(max_iters)
            (i >= max_iters) && break
        end
        if !isnothing(target_err)
            (err < target_err) && break
        end
        i += 1
    end

    return X, Y, plots, errs
end

function model_predict_MC(model_predict, x; param_samples, ϵ_samples, noise, sample_count)
    val = 0.
    for i in 1:sample_count
        val += model_predict(x, param_samples[i,:]...) + (noise * ϵ_samples[i])
    end
    return val / sample_count
end

function opt_acquisition(acq, domain_lb, domain_ub; multistart=1)
    dim = length(domain_lb)
    starts = rand(dim, multistart) .* (domain_ub .- domain_lb) .+ domain_lb
    best_res = (0., -Inf)
    for i in 1:multistart
        s = starts[:,i]
        opt_res = Optim.optimize(x -> -acq(x), domain_lb, domain_ub, s, Fminbox(LBFGS()))
        res = Optim.minimizer(opt_res), -Optim.minimum(opt_res)
        (res[2] > best_res[2]) && (best_res = res)
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
