module Boss

using Plots
using LinearAlgebra
using Distributions
using Turing
using FLoops

include("model.jl")
include("gp.jl")
include("acq.jl")
include("plotting.jl")
include("utils.jl")

export boss
export LinFitness, NonlinFitness, LinModel, NonlinModel

const ModelPost = Tuple{Union{Function, Nothing}, Union{Function, Nothing}}

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
    mc_sample_count=4000,
    acq_opt_multistart=200,
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
    kwargs...
)
    # - - - - - - - - INITIALIZATION - - - - - - - - - - - - - - - -
    isnothing(max_iters) && isnothing(target_err) && throw(ArgumentError("No termination condition provided. Use kwargs 'max_iters' or 'target_err' to define a termination condition."))

    lin_model = (model isa LinModel)
    feasibility = !isnothing(Z)
    feasibility_count = feasibility ? size(Z)[2] : 0
    init_data_size = size(X)[1]
    y_dim = size(Y)[2]
    x_dim = size(X)[2]
    isnothing(gp_params_priors) && (gp_params_priors = [MvLogNormal(ones(x_dim), ones(x_dim)) for _ in 1:y_dim])
    isnothing(feasibility_gp_params_priors) && (feasibility_gp_params_priors = [MvLogNormal(ones(x_dim), ones(x_dim)) for _ in 1:feasibility_count])

    Φs = lin_model ? init_Φs(model.lift, X) : nothing
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

        # parametric model
        if plot_all_models || (use_model == :param) || (use_model == :semiparam)
            if lin_model
                # TODO refactor needed
                throw(ErrorException("Support for linear models broken. Needs refactoring."))
                param_posts_ = [lin_param_posterior(Φs[i], Y[:,i], model.param_priors[i], noise_priors[i]) for i in 1:y_dim]
                parametric = (x->lin_model_predict(x, model.lift, getindex.(param_posts_, 1)), x->lin_model_vars(x, model.lift, getindex.(param_posts_, 2), noise_priors))
            else
                par_params_samples, par_noise_samples = sample_param_posterior(model.prob_model, noise_priors, X, Y; y_dim, param_count=model.param_count, sample_count=mc_sample_count)
                parametric = fit_par_mc(model.predict, par_params_samples, par_noise_samples; sample_count=mc_sample_count)
            end
        else
            parametric = (nothing, nothing)
        end

        # semiparametric model (param + GP)
        if plot_all_models || (use_model == :semiparam)
            if gp_hyperparam_alg == :LBFGS
                params_, noise_ = opt_gp_posterior(X, Y, gp_params_priors, noise_priors; y_dim, mean=parametric[1], kernel, multistart=mc_sample_count, info, debug)
                semiparametric = fit_gps(X, Y, params_, noise_; y_dim, mean=parametric[1], kernel)
            elseif gp_hyperparam_alg == :NUTS
                param_samples_, noise_samples_ = sample_gp_posterior(X, Y, gp_params_priors, noise_priors; x_dim, y_dim, mean=parametric[1], kernel, sample_count=mc_sample_count)
                semiparametric = fit_gps_mc(X, Y, param_samples_, noise_samples_; y_dim, mean=parametric[1], kernel, sample_count=mc_sample_count)
            end
        else
            semiparametric = (nothing, nothing)
        end

        # nonparametric model (GP)
        if plot_all_models || (use_model == :nonparam)
            if gp_hyperparam_alg == :LBFGS
                params_, noise_ = opt_gp_posterior(X, Y, gp_params_priors, noise_priors; y_dim, kernel, multistart=mc_sample_count, info, debug)
                nonparametric = fit_gps(X, Y, params_, noise_; y_dim, kernel)
            elseif gp_hyperparam_alg == :NUTS
                params_samples_, noise_samples_ = sample_gp_posterior(X, Y, gp_params_priors, noise_priors; x_dim, y_dim, kernel, sample_count=mc_sample_count)
                nonparametric = fit_gps_mc(X, Y, params_samples_, noise_samples_; y_dim, kernel, sample_count=mc_sample_count)
            end
        else
            nonparametric = (nothing, nothing)
        end

        # feasibility models (GPs)
        if feasibility
            # TODO modify the prior mean ?
            # TODO provide option for defining semiparametric models for feasibility constraints ?
            if gp_hyperparam_alg == :LBFGS
                params_, noise_ = opt_gp_posterior(X, Z, feasibility_gp_params_priors, feasibility_noise_priors; y_dim=feasibility_count, kernel=feasibility_kernel, multistart=mc_sample_count, info, debug)
                feasibility_model = fit_gps(X, Z, params_, noise_; y_dim=feasibility_count, kernel=feasibility_kernel)
            elseif gp_hyperparam_alg == :NUTS
                param_samples_, noise_samples_ = sample_gp_posterior(X, Z, feasibility_gp_params_priors, feasibility_noise_priors; x_dim, y_dim=feasibility_count, kernel=feasibility_kernel, sample_count=mc_sample_count)
                feasibility_model = fit_gps_mc(X, Z, param_samples_, noise_samples_; y_dim=feasibility_count, kernel=feasibility_kernel, sample_count=mc_sample_count)
            end
        else
            feasibility_model = nothing
        end

        # - - - - - - - - UTILITY MAXIMIZATION - - - - - - - - - - - - - - - -
        info && print("  optimizing utility ...\n")
        multistart = acq_opt_multistart
        res_param = res_semiparam = res_nonparam = nothing
        acq_param = acq_semiparam = acq_nonparam = nothing

        ϵ_samples = rand(Distributions.Normal(), (mc_sample_count, y_dim))
        
        # parametric
        if plot_all_models || (use_model == :param)
            ei_par_ = x->EI(x, fitness, parametric, ϵ_samples; best_yet=last(bsf), sample_count=mc_sample_count)
            acq_par = construct_acq(ei_par_, feasibility_model; feasibility, best_yet=last(bsf))
            res_par = opt_acq(acq_par, domain; multistart, info, debug)
        end

        # semiparametric
        if plot_all_models || (use_model == :semiparam)
            ei_semipar_ = x->EI(x, fitness, semiparametric, ϵ_samples; best_yet=last(bsf), sample_count=mc_sample_count)
            acq_semipar = construct_acq(ei_semipar_, feasibility_model; feasibility, best_yet=last(bsf))
            res_semipar = opt_acq(acq_semipar, domain; multistart, info, debug)
        end

        # nonparametric
        if plot_all_models || (use_model == :nonparam)
            ei_nonpar_ = x->EI(x, fitness, nonparametric, ϵ_samples; best_yet=last(bsf), sample_count=mc_sample_count)
            acq_nonpar = construct_acq(ei_nonpar_, feasibility_model; feasibility, best_yet=last(bsf))
            res_nonpar = opt_acq(acq_nonpar, domain; multistart, info, debug)
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
                feasibility_model,
                X, Y;
                iter,
                y_dim,
                feasibility,
                feasibility_count,
                domain,
                init_data_size,
                show_plots,
                kwargs...
            )
            append!(plots, ps)
        end

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

    if feasible_ && (f_ > last(bsf))
        push!(bsf, f_)
    else
        push!(bsf, last(bsf))
    end

    return X, Y, Z, Φs, F, bsf
end

# Sample from the posterior parameter distributions given the data 'X', 'Y'.
function sample_param_posterior(prob_model, noise_priors, X, Y; y_dim, param_count, sample_count)
    chain = Turing.sample(prob_model(X, Y, noise_priors), NUTS(), sample_count; verbose=false)
    params = reduce(hcat, [chain[Symbol("params[$i]")][:] for i in 1:param_count])
    noise = reduce(hcat, [chain[Symbol("noise[$i]")][:] for i in 1:y_dim])
    return params, noise
end

function sample_gp_posterior(X, Y, params_priors, noise_priors; x_dim, y_dim, mean=x->zeros(y_dim), kernel, sample_count)
    samples = [gp_sample_params_nuts(X, Y[:,i], params_priors[i], noise_priors[i]; x_dim, mean=x->mean(x)[i], kernel, sample_count) for i in 1:y_dim]
    params = [s[1] for s in samples]
    noise = [s[2] for s in samples]
    return params, noise
end

function opt_gp_posterior(X, Y, params_priors, noise_priors; y_dim, mean=x->zeros(y_dim), kernel, multistart, info, debug)
    P = [gp_fit_params_lbfgs(X, Y[:,i], params_priors[i], noise_priors[i]; mean=x->mean(x)[i], kernel, multistart, info, debug) for i in 1:y_dim]
    params = [p[1] for p in P]
    noise = [p[2] for p in P]
    return params, noise
end

function fit_gps_mc(X, Y, param_samples, noise_samples; y_dim, mean=x->zeros(y_dim), kernel, sample_count)
    gp_preds = [[gp_pred_distr(X, Y[:,i], param_samples[i][s,:], noise_samples[i][s]; mean=x->mean(x)[i], kernel) for s in 1:sample_count] for i in 1:y_dim]
    gp_model = (x -> [Distributions.mean([pred[1](x) for pred in gp_preds[i]]) for i in 1:y_dim],
                x -> [Distributions.mean([pred[2](x) for pred in gp_preds[i]]) for i in 1:y_dim])
    return gp_model
end

function fit_gps(X, Y, params, noise; y_dim, mean=x->zeros(y_dim), kernel)
    gp_preds = [gp_pred_distr(X, Y[:,i], params[i], noise[i]; mean=x->mean(x)[i], kernel) for i in 1:y_dim]
    gp_model = (x -> [pred[1](x) for pred in gp_preds],
                x -> [pred[2](x) for pred in gp_preds])
    return gp_model
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

function lin_model_predict(x, lift, μθs)
    return transpose.(μθs) .* lift(x)
end

function lin_model_vars(x, lift, Σθs, noise)
    ϕs = lift(x)
    return Diagonal(noise .+ (transpose.(ϕs) .* Σθs .* ϕs))
end

function fit_par_mc(model_predict, param_samples, noise_samples; sample_count)
    noise = mean(noise_samples; dims=1)[:]
    return x -> sum([model_predict(x, param_samples[i,:]) for i in 1:sample_count]) ./ sample_count,
           x -> noise
end

# TODO refactor for noise prior instead of a given noise value
function lin_param_posterior(Φ, Y_col, param_prior, noise)
    ω = 1 / noise
    μθ, Σθ = param_prior
    inv_Σθ = inv(Σθ)

    Σθ_ = inv(inv_Σθ + ω * Φ' * Φ)
    μθ_ = Σθ_ * (inv_Σθ * μθ + ω * Φ' * Y_col)

    return μθ_, Σθ_
end

function feasibility_probabilities(feasibility_model)
    function p(x)
        μ, σ = feasibility_model[1](x), feasibility_model[2](x)
        N = length(μ)
        distrs = [Distributions.Normal(μ[i], σ[i]) for i in 1:N]
        return [(1. - cdf(d, 0.)) for d in distrs]
    end
    return p
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
