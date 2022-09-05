module Boss

using Logging
using Plots
using LinearAlgebra
using Distributions

include("acq.jl")
include("model.jl")
include("gp.jl")
include("semiparam.jl")
include("utils.jl")
include("plotting.jl")

export boss
export LinFitness, NonlinFitness, LinModel, NonlinModel
export MCSettings

const ModelPost = Tuple{Union{Function, Nothing}, Union{Function, Nothing}}

# TODO refactor: add typing, multiple dispatch, ...
# TODO docs, comments & example usage
# TODO refactor model error computation
# TODO add param & return types
# TODO input/output space normalization


"""
Bayesian optimization with a N->N dimensional semiparametric surrogate model.

The algorithm performs most expensive computations in parallel.
Make sure you have set the 'JULIA_NUM_THREDS' environment variable correctly.



# Without feasibility constraints (only input domain constraints):

To maximize 'fitness(f(x))' such that 'x ∈ domain' call:
```julia-repl
X, Y, bsf, errs, plots = boss(f, fitness, X, Y, model, domain; kwargs...);
```

## Parameters:

- f:           The objective function 'f: x -> y'. Takes vector x, returns vector y.
 
- fitness:     Fitness function 'fitness: y -> Float' given as an instance of 'Boss.Fitness' or as a Function.
 
- X:           Matrix containing the previously evaluated input points 'x' as its rows.
 
- Y:           Matrix containing the output vectors 'y' of previous evaluations of the objective function as its rows.
 
- model:       The parametric model used as the prior mean of the semiparametric model given as an instance of 'Boss.ParamModel'.
               The use of 'Boss.LinModel' will allow for more efficient analytical computations in the future. (Not implemented yet.)
 
- domain:      The input domain of the objective function given as a Tuple of lower and upper bound vectors
                or as an instance of 'Optim.TwiceDifferentiableConstraints' for more complex input constraints.

## Outputs:

- bsf:         Vector containing the history of the best-so-far found fitness in each iteration.

- errs:        Vector containing the history of the model RMS error.
                Returns 'nothing' if no test data have been provided.

- plots:       Vector of 'Plots.Plot' of each iteration of the algorithm.
                Returns 'nothing' unless the kwarg 'make_plots' is set to true.



# With feasibility constraints on the output 'y':

To maximize 'fitness(f(x))' such that 'gᵢ(x) > 0' and 'x ∈ domain' call:
```julia-repl
X, Y, Z, bsf, errs, plots = boss(fg, fitness, X, Y, Z, model, domain; kwargs...);
```

## Parameters:

- fg:          Function 'fg: x -> (f(x), [gᵢ(x) for ∀i])'
                where 'f: x -> y' is the objective function
                and 'gᵢ: x -> zᵢ' are the feasibility constraints.
                The return value should be a Tuple of two vectors.

- Z:           Matrix containing the output values 'zᵢ' of previous evaluations of the feasibility constraints.



# Keyword arguments:

## Required kwargs:

- noise_priors:                 Vector of distributions describing the prior belief about the evaluation noise of each output dimension.

- feasibility_noise_priors:     Vector of distributions describing the prior belief about the evaluation noise of each feasibility constraint.
                                Only required if feasibility constraints are provided.

## Termination conditions:

At least one of the termination conditions has to be provided.

- max_iters:                    The number of iterations after which the algorithm stops.
                                Equal to the number of objective function evaluations.
                                If it is set to zero, the models will be fitted and the objective function will not be evaluated.

- target_error:                 !!! Currently disabled.
                                The target RMS error of the model. The algorithm stops when the model error is lower than this value.
                                The kwargs 'test_X' and 'test_Y' containing the test data have to be provided if this termination condition is used.

## Optional kwargs:

- mc_settings:                  An instance of 'Boss.MCSettings' defining the hyperparameters of the MC sampler.

- acq_opt_multistart:           Defines how many times is the acquisition function optimization restarted to find the global optimum.

- param_opt_multistart:         Defines how many times is the model parameter optimization restarted to find the global optimum.

- gp_params_priors:             The prior distributions of the GP hyperparameters.

- feasibility_gp_params_priors  The prior distributions of the hyperparameters of the GPs used to model the feasibility constraints.

- info:                         Set to false to disable the info prints.

- debug:                        Set to true to stop the algorithm on any optimization error.

- make_plots:                   Set to true to generate plots.

- show_plots:                   Set to false to not display plots as they are generated.

- plot_all_models:              Set to true to fit and plot all models (not only the one used by the algorithm).
                                Significantly slows down the algorithm!

- f_true:                       The objective function 'f: x -> y' without noise. (For plotting purposes only.)

- kernel:                       The kernel used in the GP models. See 'AbstractGPs.jl' for more info.

- feasibility_kernel:           The kernel used in the GPs used to model the feasibility constraints. See 'AbstractGPs.jl' for more info.

- use_model:                    Defines which surrogate model type is to be used by the algorithm.
                                Possible values are ':param, :semiparam, :nonparam' for the parametric, semiparametric or nonparametric models.

- param_fit_alg:                Defines which algorithm is used to fit the parameters of the surrogate model.
                                Possible values are: ':LBFGS' for the MLE via the LBFGS optimization algorithm,
                                                     ':NUTS' for the MC sampling of the hyperparameter posterior via the NUTS algorithm.

- feasibility_param_fit_alg:    Defines which algorithm is used to fit the parameters of the the feasibility constraint models.

## Other kwargs:

- kwargs...:        Additional kwargs are forwarded to plotting.

"""
function boss(f, fitness, X, Y, model::ParamModel, domain; kwargs...)
    fg(x) = f(x), Float64[]
    return boss(fg, fitness, X, Y, nothing, model, domain; kwargs...)
end

function boss(fg, fitness::Function, X, Y, Z, model::ParamModel, domain; kwargs...)
    fit = NonlinFitness(fitness)
    return boss(fg, fit, X, Y, Z, model, domain; kwargs...)
end

function boss(fg::Function, fitness::Fitness, X, Y, Z, model::ParamModel, domain;
    noise_priors,
    feasibility_noise_priors=nothing,
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
    param_fit_alg=:NUTS,  # :NUTS, :LBFGS
    feasibility_param_fit_alg=:LBFGS,  # :NUTS, :LBFGS
    kwargs...
)
    # - - - - - - - - INITIALIZATION - - - - - - - - - - - - - - - -
    isnothing(max_iters) && isnothing(target_err) && throw(ArgumentError("No termination condition provided. Use kwargs 'max_iters' or 'target_err' to define a termination condition."))

    # Workaround: https://github.com/TuringLang/Turing.jl/issues/1398
    if debug
        Logging.disable_logging(Logging.BelowMinLevel)
    else
        Logging.disable_logging(Logging.Warn)
    end

    feasibility = !isnothing(Z)
    feasibility_count = feasibility ? size(Z)[2] : 0
    init_data_size = size(X)[1]
    y_dim = size(Y)[2]
    x_dim = size(X)[2]
    isnothing(gp_params_priors) && (gp_params_priors = [MvLogNormal(ones(x_dim), ones(x_dim)) for _ in 1:y_dim])
    isnothing(feasibility_gp_params_priors) && (feasibility_gp_params_priors = [MvLogNormal(ones(x_dim), ones(x_dim)) for _ in 1:feasibility_count])

    Φs = (model isa LinModel) ? init_Φs(model.lift, X) : nothing
    F = [fitness(y) for y in eachrow(Y)]
    bsf = Union{Nothing, Float64}[get_best_yet(F, X, Z, domain; data_size=init_data_size)]

    plots = make_plots ? Plots.Plot[] : nothing
    errs = (isnothing(test_X) || isnothing(test_Y)) ? nothing : Vector{Float64}[]

    # TODO refactor model error computation
    errs = nothing  # remove

    if model isa LinModel
        # TODO implement analytical calculations for linear models
        model = convert(NonlinModel, model)
    end

    # - - - - - - - - MAIN OPTIMIZATION LOOP - - - - - - - - - - - - - - - -
    iter = 0
    opt_new_x = Float64[]
    while true
        info && print("\nITER $iter\n")

        # - - - - - - - - NEW DATA - - - - - - - - - - - - - - - -
        if iter != 0
            info && print("  evaluating the objective function ...\n")
            X, Y, Z, Φs, F, bsf = augment_data!(opt_new_x, fg, model, fitness, domain, X, Y, Z, Φs, F, bsf; feasibility, y_dim, info)
        end

        # - - - - - - - - MODEL INFERENCE - - - - - - - - - - - - - - - -
        info && print("  model inference ...\n")

        # PARAMETRIC MODEL
        if (make_plots && plot_all_models) || (use_model == :param)
            if param_fit_alg == :LBFGS
                par_params, par_noise = fit_model_params_lbfgs(X, Y, model, noise_priors; y_dim, multistart=param_opt_multistart, info, debug)
                parametric = (x -> model(x, par_params),
                              x -> par_noise)
                par_models = nothing
            
            elseif param_fit_alg == :NUTS
                par_param_samples, par_noise_samples = sample_param_posterior(X, Y, model, noise_priors; y_dim, mc_settings)
                par_models = [(x->model(x, par_param_samples[s]), x->par_noise_samples[s]) for s in 1:sample_count(mc_settings)]

                if make_plots
                    loglikes_ = [loglike(m, X, Y; data_size=size(X)[1]) for m in par_models]
                    parametric = par_models[argmax(loglikes_)]
                    samples_lable = "param samples"
                else
                    parametric = (nothing, nothing)
                end
            end
        else
            parametric = (nothing, nothing)
            par_models = nothing
        end

        # SEMIPARAMETRIC MODEL (param + GP)
        if (make_plots && plot_all_models) || (use_model == :semiparam)
            if param_fit_alg == :LBFGS
                semipar_mean_params, semipar_params, semipar_noise = opt_semipar_posterior(X, Y, model, gp_params_priors, noise_priors; x_dim, y_dim, kernel, multistart=param_opt_multistart, info, debug)
                semipar_mean(x) = model(x, semipar_mean_params)
                semiparametric = fit_gps(X, Y, semipar_params, semipar_noise; y_dim, mean=semipar_mean, kernel)
            
            elseif param_fit_alg == :NUTS
                semipar_mean_param_samples, semipar_param_samples, semipar_noise_samples = sample_semipar_posterior(X, Y, model, gp_params_priors, noise_priors; x_dim, y_dim, kernel, mc_settings)
                semipar_models = [fit_gps(X, Y, semipar_param_samples[s], semipar_noise_samples[s]; y_dim, mean=x->model(x, semipar_mean_param_samples[s]), kernel) for s in 1:sample_count(mc_settings)]
                semiparametric = (nothing, nothing)

                if make_plots
                    if isnothing(par_models)
                        par_models = [(x->model(x, semipar_mean_param_samples[s]), nothing) for s in 1:sample_count(mc_settings)]
                        samples_lable = "mean samples"
                    end
                end
            end
        else
            semiparametric = (nothing, nothing)
        end

        # NONPARAMETRIC MODEL (GP)
        if (make_plots && plot_all_models) || (use_model == :nonparam)
            if param_fit_alg == :LBFGS
                nonpar_params, nonpar_noise = opt_gp_posterior(X, Y, gp_params_priors, noise_priors; y_dim, kernel, multistart=param_opt_multistart, info, debug)
                nonparametric = fit_gps(X, Y, nonpar_params, nonpar_noise; y_dim, kernel)
            
            elseif param_fit_alg == :NUTS
                nonpar_param_samples, nonpar_noise_samples = sample_gp_posterior(X, Y, gp_params_priors, noise_priors; x_dim, y_dim, kernel, mc_settings)
                nonpar_models = [fit_gps(X, Y, nonpar_param_samples[s], nonpar_noise_samples[s]; y_dim, kernel) for s in 1:sample_count(mc_settings)]
                nonparametric = (nothing, nothing)
            end
        else
            nonparametric = (nothing, nothing)
        end

        # feasibility models (GPs)
        if feasibility
            if feasibility_param_fit_alg == :LBFGS
                feas_params, feas_noise = opt_gp_posterior(X, Z, feasibility_gp_params_priors, feasibility_noise_priors; y_dim=feasibility_count, kernel=feasibility_kernel, multistart=param_opt_multistart, info, debug)
                model_ = fit_gps(X, Z, feas_params, feas_noise; y_dim=feasibility_count, kernel=feasibility_kernel)
                feas_probs = x->feasibility_probabilities(model_)(x)
            
            elseif feasibility_param_fit_alg == :NUTS
                feas_param_samples, feas_noise_samples = sample_gp_posterior(X, Z, feasibility_gp_params_priors, feasibility_noise_priors; x_dim, y_dim=feasibility_count, kernel=feasibility_kernel, mc_settings)
                feas_models = [fit_gps(X, Z, feas_param_samples[s], feas_noise_samples[s]; y_dim=feasibility_count, kernel=feasibility_kernel) for s in 1:sample_count(mc_settings)]
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
        if (make_plots && plot_all_models) || (use_model == :param)
            if param_fit_alg == :LBFGS
                ei_par_ = x->EI(x, fitness, parametric, ϵ_samples; best_yet=last(bsf), sample_count=sample_count(mc_settings))
            elseif param_fit_alg == :NUTS
                ei_par_ = x->mean([EI(x, fitness, m, ϵ_samples; best_yet=last(bsf), sample_count=sample_count(mc_settings)) for m in par_models])
            end
            acq_par = construct_acq(ei_par_, feas_probs; feasibility, best_yet=last(bsf))
            res_par = opt_acq(acq_par, domain; x_dim, multistart=acq_opt_multistart, info, debug)
        else
            acq_par, res_par = nothing, nothing
        end

        # semiparametric
        res_semipar = nothing
        if (make_plots && plot_all_models) || (use_model == :semiparam)
            if param_fit_alg == :LBFGS
                ei_semipar_ = x->EI(x, fitness, semiparametric, ϵ_samples; best_yet=last(bsf), sample_count=sample_count(mc_settings))
            elseif param_fit_alg == :NUTS
                ei_semipar_ = x->mean([EI(x, fitness, m, ϵ_samples; best_yet=last(bsf), sample_count=sample_count(mc_settings)) for m in semipar_models])
            end
            acq_semipar = construct_acq(ei_semipar_, feas_probs; feasibility, best_yet=last(bsf))
            res_semipar = opt_acq(acq_semipar, domain; x_dim, multistart=acq_opt_multistart, info, debug)
        else
            acq_semipar, res_semipar = nothing, nothing
        end

        # nonparametric
        res_nonpar = nothing
        if (make_plots && plot_all_models) || (use_model == :nonparam)
            if param_fit_alg == :LBFGS
                ei_nonpar_ = x->EI(x, fitness, nonparametric, ϵ_samples; best_yet=last(bsf), sample_count=sample_count(mc_settings))
            elseif param_fit_alg == :NUTS
                ei_nonpar_ = x->mean([EI(x, fitness, m, ϵ_samples; best_yet=last(bsf), sample_count=sample_count(mc_settings)) for m in nonpar_models])
            end
            acq_nonpar = construct_acq(ei_nonpar_, feas_probs; feasibility, best_yet=last(bsf))
            res_nonpar = opt_acq(acq_nonpar, domain; x_dim, multistart=acq_opt_multistart, info, debug)
        else
            acq_nonpar, res_nonpar = nothing, nothing
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
                param_fit_alg,
                samples_lable,
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

function augment_data!(opt_new_x, fg, model::ParamModel, fitness::Fitness, domain, X, Y, Z, Φs, F, bsf; feasibility, y_dim, info)
    x_ = opt_new_x
    y_, z_ = fg(x_)
    f_ = fitness(y_)

    info && print("  new data-point: x = $x_\n"
                * "                  y = $y_\n"
                * "                  f = $f_\n")
    X = vcat(X, x_')
    Y = vcat(Y, [y_...]')
    F = vcat(F, f_)

    # TODO
    # if model isa LinModel
    #     ϕs = model.lift(x_)
    #     for i in 1:y_dim
    #         Φs[i] = vcat(Φs[i], ϕs[i]')
    #     end
    # end

    in_domain_ = is_in_domain(x_, domain)
    
    if feasibility
        feasible_ = is_feasible(z_)
        info && (feasible_ ? print("                  feasible\n") : print("                  infeasible\n"))
        Z = vcat(Z, [z_...]')
    else
        feasible_ = true
    end

    if in_domain_ && feasible_ && (isnothing(last(bsf)) || (f_ > last(bsf)))
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

function get_best_yet(F, X, Z, domain; data_size)
    in_domain = get_in_domain(X, domain; data_size)
    feasible = get_feasible(Z; data_size)
    good = in_domain .& feasible
    any(good) || return nothing
    return maximum([F[i] for i in 1:data_size if good[i]])
end

function get_in_domain(X, domain; data_size)
    return [is_in_domain(X[i,:], domain) for i in 1:data_size]
end

function is_in_domain(x, domain::Tuple)
    lb, ub = domain
    any(x .< lb) && return false
    any(x .> ub) && return false
    return true
end

function is_in_domain(x, domain::TwiceDifferentiableConstraints)
    return Optim.isinterior(domain, x)
end

function get_feasible(Z; data_size)
    isnothing(Z) && return [false for i in 1:data_size]
    return [is_feasible(Z[i,:]) for i in 1:data_size]
end

function is_feasible(z)
    return all(z .>= 0)
end

function loglike(model, X, Y; data_size)
    pred_distr(x) = MvNormal(model[1](x), model[2](x))
    return sum([logpdf(pred_distr(X[i,:]), Y[i,:]) for i in 1:data_size])
end

end  # module
