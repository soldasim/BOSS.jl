module Boss

# Workaround: https://github.com/TuringLang/Turing.jl/issues/1398
using Logging
Logging.disable_logging(Logging.Info)
Logging.disable_logging(Logging.Warn)

using Plots
using LinearAlgebra
using Distributions
using FLoops

include("model.jl")
include("gp.jl")
include("acq.jl")
include("utils.jl")
include("plotting.jl")

export boss
export LinFitness, NonlinFitness, LinModel, NonlinModel
export MCSettings

const ModelPost = Tuple{Union{Function, Nothing}, Union{Function, Nothing}}

# TODO Try out param model fitting via NUTS sampling.
# TODO refactor: add typing, multiple dispatch, ...
# TODO docs, comments & example usage
# TODO refactor model error computation
# TODO add param & return types

# Bayesian optimization with a N->N dimensional semiparametric surrogate model.
# Gaussian noise only. (for now)
# Plotting only works for 1D->1D problems.
#
# w/o feasibility -> boss(f, X, Y, model, domain; kwargs...)
#                 ->   f(x) = y
#
# w/  feasibility -> boss(f, X, Y, Z, model, domain; kwargs...)
#                 ->   f(x) = (y, z)

function boss(f, fitness, X, Y, model::ParamModel, domain; kwargs...)
    fg(x) = f(x), Float64[]
    return boss(fg, fitness, X, Y, nothing, model, domain; kwargs...)
end

function boss(fg, fitness::Function, X, Y, Z, model::ParamModel, domain; kwargs...)
    fit = NonlinFitness(fitness)
    return boss(fg, fit, X, Y, Z, model, domain; kwargs...)
end

function boss(fg, fitness::Fitness, X, Y, Z, model::ParamModel, domain;
    noise_priors,
    feasibility_noise_priors,
    mc_settings=MCSettings(400, 20, 8, 6),
    acq_opt_multistart=80,
    param_opt_multistart=80,
    gp_params_priors=nothing,
    feasibility_gp_params_priors=nothing,
    info=true,
    debug=false,  # stop on exception
    make_plots=false,
    show_plots=true,
    plot_all_models=false,
    f_true=nothing,
    max_iters=nothing,
    test_X=nothing,
    test_Y=nothing,
    target_err=nothing,
    kernel=Matern52Kernel(),
    feasibility_kernel=kernel,
    use_model=:semiparam,  # :param, :semiparam, :nonparam
    gp_hyperparam_alg=:NUTS,  # :NUTS, :LBFGS
    feasibility_gp_hyperparam_alg=:LBFGS,  # :NUTS, :LBFGS
    kwargs...
)
    # - - - - - - - - INITIALIZATION - - - - - - - - - - - - - - - -
    isnothing(max_iters) && isnothing(target_err) && throw(ArgumentError("No termination condition provided. Use kwargs 'max_iters' or 'target_err' to define a termination condition."))

    feasibility = !isnothing(Z)
    feasibility_count = feasibility ? size(Z)[2] : 0
    init_data_size = size(X)[1]
    y_dim = size(Y)[2]
    x_dim = size(X)[2]
    isnothing(gp_params_priors) && (gp_params_priors = [MvLogNormal(ones(x_dim), ones(x_dim)) for _ in 1:y_dim])
    isnothing(feasibility_gp_params_priors) && (feasibility_gp_params_priors = [MvLogNormal(ones(x_dim), ones(x_dim)) for _ in 1:feasibility_count])

    Φs = (model isa LinModel) ? init_Φs(model.lift, X) : nothing
    F = [fitness(y) for y in eachrow(Y)]
    bsf = [get_best_yet(F, Z; data_size=init_data_size)]

    plots = make_plots ? Plots.Plot[] : nothing
    errs = (isnothing(test_X) || isnothing(test_Y)) ? nothing : Vector{Float64}[]

    # - - - - - - - - MAIN OPTIMIZATION LOOP - - - - - - - - - - - - - - - -
    iter = 0
    opt_new_x = Float64[]
    while true
        info && print("\nITER $iter\n")

        # - - - - - - - - NEW DATA - - - - - - - - - - - - - - - -
        if iter != 0
            info && print("  evaluating the objective function ...\n")
            X, Y, Z, Φs, F, bsf = augment_data!(opt_new_x, fg, model, fitness, X, Y, Z, Φs, F, bsf; feasibility, y_dim, info)
        end

        # - - - - - - - - MODEL INFERENCE - - - - - - - - - - - - - - - -
        info && print("  model inference ...\n")

        # PARAMETRIC MODEL
        if model isa LinModel
            # TODO
            throw(ErrorException("Support for linear models not implemented yet."))
            # param_post_ = param_posterior(Φs, Y, model, noise_priors)
            # parametric = (...)
        end

        if plot_all_models || (use_model == :param)
            # NUTS - model posterior samples (for par acq)
            par_param_samples, par_noise_samples = sample_param_posterior(X, Y, model, noise_priors; y_dim, mc_settings)
            par_models = [(x->model(x, par_param_samples[i]), x->par_noise_samples[i]) for i in 1:sample_count(mc_settings)]
        end

        if plot_all_models || (use_model == :param) || (use_model == :semiparam)
            # LBFGS - maximum likelihood fit (for semipar model mean and plotting)
            par_params, par_noise = fit_model_params_lbfgs(X, Y, model, noise_priors; y_dim, multistart=param_opt_multistart, info, debug)
            parametric = (x -> model.predict(x, par_params),
                            x -> par_noise)
        else
            parametric = (nothing, nothing)
        end

        # SEMIPARAMETRIC MODEL (param + GP)
        if plot_all_models || (use_model == :semiparam)
            if gp_hyperparam_alg == :LBFGS
                semi_params, semi_noise = opt_gp_posterior(X, Y, gp_params_priors, noise_priors; y_dim, mean=parametric[1], kernel, multistart=param_opt_multistart, info, debug)
                semiparametric = fit_gps(X, Y, semi_params, semi_noise; y_dim, mean=parametric[1], kernel)
            elseif gp_hyperparam_alg == :NUTS
                semipar_param_samples, semipar_noise_samples = sample_gp_posterior(X, Y, gp_params_priors, noise_priors; x_dim, y_dim, mean=parametric[1], kernel, mc_settings)
                semipar_models = fit_gp_samples(X, Y, semipar_param_samples, semipar_noise_samples; y_dim, mean=parametric[1], kernel, sample_count=sample_count(mc_settings))
                
                # For plotting only:
                # semiparametric = (x -> mean([m[1](x) for m in semipar_models]),
                #                   x -> mean([m[2](x) for m in semipar_models]))
                semiparametric = last(semipar_models)
            end
        else
            semiparametric = (nothing, nothing)
        end

        # NONPARAMETRIC MODEL (GP)
        if plot_all_models || (use_model == :nonparam)
            if gp_hyperparam_alg == :LBFGS
                nonpar_params, nonpar_noise = opt_gp_posterior(X, Y, gp_params_priors, noise_priors; y_dim, kernel, multistart=param_opt_multistart, info, debug)
                nonparametric = fit_gps(X, Y, nonpar_params, nonpar_noise; y_dim, kernel)
            elseif gp_hyperparam_alg == :NUTS
                nonpar_param_samples, nonpar_noise_samples = sample_gp_posterior(X, Y, gp_params_priors, noise_priors; x_dim, y_dim, kernel, mc_settings)
                nonpar_models = fit_gp_samples(X, Y, nonpar_param_samples, nonpar_noise_samples; y_dim, kernel, sample_count=sample_count(mc_settings))
                
                # For plotting only:
                # nonparametric = (x -> mean([m[1](x) for m in nonpar_models]),
                #                  x -> mean([m[2](x) for m in nonpar_models]))
                nonparametric = last(nonpar_models)
            end
        else
            nonparametric = (nothing, nothing)
        end

        # feasibility models (GPs)
        if feasibility
            # TODO modify the prior mean ?
            # TODO provide option for defining semiparametric models for feasibility constraints ?
            if feasibility_gp_hyperparam_alg == :LBFGS
                feas_params, feas_noise = opt_gp_posterior(X, Z, feasibility_gp_params_priors, feasibility_noise_priors; y_dim=feasibility_count, kernel=feasibility_kernel, multistart=param_opt_multistart, info, debug)
                model_ = fit_gps(X, Z, feas_params, feas_noise; y_dim=feasibility_count, kernel=feasibility_kernel)
                feas_probs = x->feasibility_probabilities(model_)(x)
            elseif feasibility_gp_hyperparam_alg == :NUTS
                feas_param_samples, feas_noise_samples = sample_gp_posterior(X, Z, feasibility_gp_params_priors, feasibility_noise_priors; x_dim, y_dim=feasibility_count, kernel=feasibility_kernel, mc_settings)
                feas_models = fit_gp_samples(X, Z, feas_param_samples, feas_noise_samples; y_dim=feasibility_count, kernel=feasibility_kernel, sample_count=sample_count(mc_settings))
                feas_probs = x->mean([feasibility_probabilities(m)(x) for m in feas_models])
            end
        else
            feas_probs = nothing
        end

        # - - - - - - - - UTILITY MAXIMIZATION - - - - - - - - - - - - - - - -
        info && print("  optimizing utility ...\n")
        ϵ_samples = rand(Distributions.Normal(), (sample_count(mc_settings), y_dim))

        # parametric
        res_par = nothing
        if plot_all_models || (use_model == :param)
            ei_par_ = x->mean([EI(x, fitness, m, ϵ_samples; best_yet=last(bsf), sample_count=sample_count(mc_settings)) for m in par_models])
            acq_par = construct_acq(ei_par_, feas_probs; feasibility, best_yet=last(bsf))
            res_par = opt_acq(acq_par, domain; x_dim, multistart=acq_opt_multistart, info, debug)
        end

        # semiparametric
        res_semipar = nothing
        if plot_all_models || (use_model == :semiparam)
            if gp_hyperparam_alg == :LBFGS
                ei_semipar_ = x->EI(x, fitness, semiparametric, ϵ_samples; best_yet=last(bsf), sample_count=sample_count(mc_settings))
            elseif gp_hyperparam_alg == :NUTS
                ei_semipar_ = x->mean([EI(x, fitness, m, ϵ_samples; best_yet=last(bsf), sample_count=sample_count(mc_settings)) for m in semipar_models])
            end
            acq_semipar = construct_acq(ei_semipar_, feas_probs; feasibility, best_yet=last(bsf))
            res_semipar = opt_acq(acq_semipar, domain; x_dim, multistart=acq_opt_multistart, info, debug)
        end

        # nonparametric
        res_nonpar = nothing
        if plot_all_models || (use_model == :nonparam)
            if gp_hyperparam_alg == :LBFGS
                ei_nonpar_ = x->EI(x, fitness, nonparametric, ϵ_samples; best_yet=last(bsf), sample_count=sample_count(mc_settings))
            elseif gp_hyperparam_alg == :NUTS
                ei_nonpar_ = x->mean([EI(x, fitness, m, ϵ_samples; best_yet=last(bsf), sample_count=sample_count(mc_settings)) for m in nonpar_models])
            end
            acq_nonpar = construct_acq(ei_nonpar_, feas_probs; feasibility, best_yet=last(bsf))
            res_nonpar = opt_acq(acq_nonpar, domain; x_dim, multistart=acq_opt_multistart, info, debug)
        end

        opt_new_x, err = select_opt_x_and_calculate_model_error(use_model, res_par, res_semipar, res_nonpar, parametric, semiparametric, nonparametric, !isnothing(errs), test_X, test_Y; info)
        isnothing(errs) || push!(errs, err)

        # - - - - - - - - PLOTTING - - - - - - - - - - - - - - - -
        if make_plots
            info && print("  plotting ...\n")
            (x_dim > 1) && throw(ErrorException("Plotting only supported for 1->N dimensional models."))
            ps = create_plots(
                f_true,
                [acq_par, acq_semipar, acq_nonpar],
                [res_par, res_semipar, res_nonpar],
                [parametric, semiparametric, nonparametric],
                par_models,
                feas_probs,
                X, Y;
                iter,
                y_dim,
                feasibility,
                feasibility_count,
                domain,
                init_data_size,
                show_plots,
                gp_hyperparam_alg,
                kwargs...
            )
            append!(plots, ps)
        end

        # TODO remove
        # skip = 5
        # for i in 1:skip:length(semipar_models)
        #     plot_model_sample(semipar_models[i], get_bounds(domain); label="semiparam", color=:purple)
        # end
        # for i in 1:skip:length(nonpar_models)
        #     plot_model_sample(nonpar_models[i], get_bounds(domain); label="nonparam", color=:blue)
        # end

        # - - - - - - - - TERMINATION CONDITIONS - - - - - - - - - - - - - - - -
        if !isnothing(max_iters)
            (iter >= max_iters) && break
        end
        if !isnothing(target_err)
            (err < target_err) && break
        end
        iter += 1
    end

    if feasibility
        return X, Y, Z, bsf, errs, plots
    else
        return X, Y, bsf, errs, plots     
    end
end

function init_Φs(lift, X)
    d = lift.(eachrow(X))
    Φs = [reduce(vcat, [ϕs[i]' for ϕs in d]) for i in 1:length(d[1])]
    return Φs
end

function augment_data!(opt_new_x, fg, model, fitness, X, Y, Z, Φs, F, bsf; feasibility, y_dim, info)
    x_ = opt_new_x
    y_, z_ = fg(x_)
    f_ = fitness(y_)

    info && print("  new data-point: x = $x_\n"
                * "                  y = $y_\n"
                * "                  f = $f_\n")
    X = vcat(X, x_')
    Y = vcat(Y, [y_...]')
    F = vcat(F, f_)

    if model isa LinModel
        ϕs = model.lift(x_)
        for i in 1:y_dim
            Φs[i] = vcat(Φs[i], ϕs[i]')
        end
    end
    
    if feasibility
        feasible_ = is_feasible(z_)
        info && (feasible_ ? print("                  feasible\n") : print("                  infeasible\n"))
        Z = vcat(Z, [z_...]')
    else
        feasible_ = true
    end

    if feasible_ && (isnothing(last(bsf)) || (f_ > last(bsf)))
        push!(bsf, f_)
    else
        push!(bsf, last(bsf))
    end

    return X, Y, Z, Φs, F, bsf
end

function select_opt_x_and_calculate_model_error(use_model, res_param, res_semiparam, res_nonparam, parametric, semiparametric, nonparametric, calc_err, test_X, test_Y; info)
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

    info && print("  optimal next x: $opt_new_x\n")
    info && print("  model error: $err\n")
    return opt_new_x, err
end

# Sample from the posterior parameter distributions given the data 'X', 'Y'.
function sample_param_posterior(X, Y, model, noise_priors; y_dim, mc_settings::MCSettings)
    param_count = model.param_count
    
    Turing.@model function prob_model(X, Y, model, noise_priors)
        params = Vector{Float64}(undef, param_count)
        for i in 1:param_count
            params[i] ~ model.param_priors[i]
        end

        noise = Vector{Float64}(undef, y_dim)
        for i in 1:y_dim
            noise[i] ~ noise_priors[i]
        end
    
        for i in 1:size(X)[1]
            Y[i,:] ~ Distributions.MvNormal(model(X[i,:], params), noise)
        end
    end

    param_symbols = vcat([Symbol("params[$i]") for i in 1:param_count],
                         [Symbol("noise[$i]") for i in 1:y_dim])
    
    samples = sample_params_nuts(prob_model(X, Y, model, noise_priors), param_symbols, mc_settings)
    params = collect(eachrow(reduce(hcat, samples[1:param_count])))
    noise = collect(eachrow(reduce(hcat, samples[param_count+1:end])))
    return params, noise
end

function opt_gp_posterior(X, Y, params_priors, noise_priors; y_dim, mean=x->zeros(y_dim), kernel, multistart, info, debug)
    P = [fit_gp_params_lbfgs(X, Y[:,i], params_priors[i], noise_priors[i]; mean=x->mean(x)[i], kernel, multistart, info, debug) for i in 1:y_dim]
    params = [p[1] for p in P]
    noise = [p[2] for p in P]
    return params, noise
end

function sample_gp_posterior(X, Y, params_priors, noise_priors; x_dim, y_dim, mean=x->zeros(y_dim), kernel, mc_settings::MCSettings)
    samples = [sample_gp_params_nuts(X, Y[:,i], params_priors[i], noise_priors[i]; x_dim, mean=x->mean(x)[i], kernel, mc_settings) for i in 1:y_dim]
    params = [(s->s[1][i,:]).(samples) for i in 1:sample_count(mc_settings)]
    noise = [(s->s[2][i]).(samples) for i in 1:sample_count(mc_settings)]
    return params, noise
end

function fit_gp_samples(X, Y, param_samples, noise_samples; y_dim, mean=x->zeros(y_dim), kernel, sample_count)
    return [fit_gps(X, Y, param_samples[i], noise_samples[i]; y_dim, mean, kernel) for i in 1:sample_count]
end

function fit_gps(X, Y, params, noise; y_dim, mean=x->zeros(y_dim), kernel)
    gp_preds = [gp_pred_distr(X, Y[:,i], params[i], noise[i]; mean=x->mean(x)[i], kernel) for i in 1:y_dim]
    return (x -> [pred[1](x) for pred in gp_preds],
            x -> [pred[2](x) for pred in gp_preds])
end

function get_best_yet(F, Z; data_size)
    feasible = get_feasible(Z; data_size)
    isempty(feasible) && return nothing
    return maximum([F[i] for i in feasible])
end

function get_feasible(Z; data_size)
    isnothing(Z) && return [i for i in 1:data_size]
    return [i for i in 1:data_size if is_feasible(Z[i,:])]
end

function is_feasible(z)
    return all(z .>= 0)
end

end  # module
