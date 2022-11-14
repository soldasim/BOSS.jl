module Boss

using Logging
using Plots
using LinearAlgebra
using Distributions
using AbstractGPs

include("utils.jl")
include("model.jl")
include("gp.jl")
include("semiparam.jl")
include("acq.jl")
include("plotting.jl")

export boss
export LinFitness, NonlinFitness, LinModel, NonlinModel
export DiscreteKernel
export MCSettings

# TODO refactor: add typing, multiple dispatch, ...
# TODO docs, comments & example usage
# TODO refactor model error computation
# TODO add param & return types
# TODO cleanup kwargs, refactor all algorithm hyperparams
# TODO input/output space normalization
# TODO change Symbol kwargs to Traits ?
# TODO cwEI can cause re-evaluation of the same points ? (e.g. when model prediction is constant)


"""
Bayesian optimization with a N->N dimensional semiparametric surrogate model.

The algorithm performs most expensive computations in parallel.
Make sure you have set the 'JULIA_NUM_THREADS' environment variable correctly.

To maximize 'fitness(f(x))' such that 'x ∈ domain' call:
```julia-repl
X, Y, bsf, errs, plots = boss(f, fitness, X, Y, model, domain; kwargs...);
```

## Parameters:

- f:           The objective function 'f: x -> y'. Takes vector x, returns vector y.
 
- fitness:     Fitness function 'fitness: y -> Float' given as an instance of 'Boss.Fitness' or as a Function.
 
- X:           Matrix containing the previously evaluated input points 'x' as columns.
 
- Y:           Matrix containing the output vectors 'y' of previous evaluations of the objective function as columns.
 
- model:       The parametric model used as the prior mean of the semiparametric model given as an instance of `Boss.ParamModel`.
               The use of `Boss.LinModel` will allow for more efficient analytical computations in the future. (Not implemented yet.)
 
- domain:      The input domain of the objective function given as a Tuple of lower and upper bound vectors
                or as an instance of `Optim.TwiceDifferentiableConstraints` for more complex input constraints.

## Outputs:

- bsf:         Vector containing the history of the best-so-far found fitness in each iteration.

- errs:        Vector containing the history of the model RMS error.
                Returns 'nothing' if no test data have been provided.

- plots:       Vector of 'Plots.Plot' of each iteration of the algorithm.
                Returns 'nothing' unless the kwarg 'make_plots' is set to true.s

# Keyword arguments:

## Required kwargs:

- noise_priors:                 Vector of distributions describing the prior belief about the evaluation noise of each output dimension.

## Termination conditions:

At least one of the termination conditions has to be provided.

- max_iters:                    The number of iterations after which the algorithm stops.
                                Equal to the number of objective function evaluations.
                                If it is set to zero, the models will be fitted and the objective function will not be evaluated.

- target_error:                 !!! Currently disabled.
                                The target RMS error of the model. The algorithm stops when the model error is lower than this value.
                                The kwargs `test_X` and `test_Y` containing the test data have to be provided if this termination condition is used.

## Optional kwargs:

- constraints:                  Can be used to specify inequality constraints on some dimensions of the objective function output space.
                                Example: `constraints=[Inf, 1000.]` specifies that the optimization is subject to y2 < 1000. and y1 ∈ R.
                                Defaults to `constraints=nothing` meaning there are no constraints on the output.

- mc_settings:                  An instance of `Boss.MCSettings` defining the hyperparameters of the MC sampler.

- vi_samples:                   Specifies how many samples are to be drawn from the hyperparameter posterior infered with VI.

- acq_opt_multistart:           Defines how many times is the acquisition function optimization restarted to find the global optimum.

- param_opt_multistart:         Defines how many times is the model parameter optimization restarted to find the global optimum.

- gp_params_priors:             The prior distributions of the GP hyperparameters.

- info:                         Set to false to disable the info prints.

- debug:                        Set to true to stop the algorithm on any optimization error.

- make_plots:                   Set to true to generate plots.

- show_plots:                   Set to false to not display plots as they are generated.

- plot_all_models:              Set to true to fit and plot all models (not only the one used by the algorithm).
                                Significantly slows down the algorithm!

- f_true:                       The objective function 'f: x -> y' without noise. (For plotting purposes only.)

- kernel:                       The kernel used in the GP models. See AbstractGPs.jl for more info.
                                The `Boss.DiscreteKernel` can be used to deal with discrete variables in the input space.

- use_model:                    Defines which surrogate model type is to be used by the algorithm.
                                Possible values are `:param`, `:semiparam`, `:nonparam` for the parametric, semiparametric or nonparametric models.

- param_fit_alg:                Defines which algorithm is used to fit the parameters of the surrogate model.
                                Possible values are: `:MLE` for the maximum likelihood estimation, 
                                                     `:BI` for Bayesian inference by approximate sampling of the posterior, 

- acq_opt_alg:                  Defines which package is used to optimize the acquisition function.
                                Possible values are `:CMAES`, `:Optim`.

- optim_options:                Options for the Optim package used to optimize the acquisition function.

- cmaes_options:                Options for the CMAES algorithm used to optimize the acquisition function.

- parallel:                     Determines whether the algorithm should use multiple threads
                                for model inference and acquisition function optimization parallelization.

## Other kwargs:

- kwargs...:        Additional kwargs are forwarded to plotting.

"""
function boss(fg::Function, fitness::Function, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, model::ParamModel, domain; kwargs...)
    fit = NonlinFitness(fitness)
    return boss(fg, fit, X, Y, model, domain; kwargs...)
end
function boss(fg::Function, fitness::Fitness, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, model::ParamModel, domain;
    noise_priors,
    constraints=nothing,
    mc_settings=MCSettings(400, 20, 8, 6),
    acq_opt_multistart=12,
    param_opt_multistart=80,
    gp_params_priors=nothing,
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
    use_model=:semiparam,  # :param, :semiparam, :nonparam
    param_fit_alg=:BI,  # :BI, :MLE
    acq_opt_alg=:CMAES,  # :CMAES, :Optim
    ϵ_sample_count=200,  # TODO refactor
    optim_options=Optim.Options(; x_abstol=1e-2, iterations=200),
    cmaes_options=Evolutionary.Options(; abstol=1e-5, iterations=800),
    parallel=true,
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

    if model isa LinModel
        # TODO implement analytical calculations for linear models
        model = convert(NonlinModel, model)
    end

    # HYPERPARAMS - - - - - - - -
    init_data_size = size(X)[2]
    y_dim = size(Y)[1]
    x_dim = size(X)[1]
    
    isnothing(gp_params_priors) && (gp_params_priors = [MvLogNormal(ones(x_dim), ones(x_dim)) for _ in 1:y_dim])
    
    if kernel isa DiscreteKernel
        discrete_dims =
            isnothing(kernel.dims) ?
            [true for _ in 1:x_dim] :
            kernel.dims
    else
        discrete_dims = [false for _ in 1:x_dim]
    end

    # DATA - - - - - - - -
    Φs = (model isa LinModel) ? init_Φs(model.lift, X) : nothing
    F = fitness.(eachcol(Y))
    bsf = Union{Nothing, Float64}[get_best_yet(F, X, Y, domain, constraints)]
    model_params_history = (use_model == :nonparam) ? nothing : Vector{Float64}[]

    plots = make_plots ? Plots.Plot[] : nothing

    # TODO refactor model error computation
    # errs = (isnothing(test_X) || isnothing(test_Y)) ? nothing : Vector{Float64}[]
    errs = nothing

    # - - - - - - - - MAIN OPTIMIZATION LOOP - - - - - - - - - - - - - - - -
    iter = 0
    opt_new_x = Float64[]
    while true
        info && print("\nITER $iter\n")

        # - - - - - - - - NEW DATA - - - - - - - - - - - - - - - -
        if iter != 0
            info && print("  evaluating the objective function ...\n")
            X, Y, Φs, F, bsf = augment_data!(opt_new_x, fg, model, fitness, domain, constraints, X, Y, Φs, F, bsf; y_dim, info)
        end

        # - - - - - - - - MODEL INFERENCE - - - - - - - - - - - - - - - -
        info && print("  model inference ...\n")
        samples_lable = nothing
        parameters = nothing  # For `model_params_history` only.  TODO: refactor

        # PARAMETRIC MODEL
        if (make_plots && plot_all_models) || (use_model == :param)
            if param_fit_alg == :MLE
                par_params, par_noise = opt_model_params(X, Y, model, noise_priors; y_dim, multistart=param_opt_multistart, parallel, info, debug)
                parametric = x -> (model(x, par_params), par_noise)
                par_models = nothing
                (use_model == :param) && (parameters = par_params)
            
            elseif param_fit_alg == :BI
                par_param_samples, par_noise_samples = sample_model_params(X, Y, model, noise_priors; y_dim, mc_settings, parallel)
                par_models = [x -> (model(x, par_param_samples[:,s]), par_noise_samples[:,s]) for s in 1:sample_count(mc_settings)]
                (use_model == :param) && (parameters = mean(eachcol(par_param_samples)))
                parametric = x -> (mapreduce(m -> m(x), .+, par_models) ./ length(par_models))  # for plotting only
            end
        else
            parametric = nothing
            par_models = nothing
        end

        # SEMIPARAMETRIC MODEL (param + GP)
        if (make_plots && plot_all_models) || (use_model == :semiparam)
            if param_fit_alg == :MLE
                semipar_mean_params, semipar_params, semipar_noise = opt_semipar_params(X, Y, model, gp_params_priors, noise_priors; x_dim, y_dim, kernel, multistart=param_opt_multistart, parallel, info, debug)
                semiparametric = gp_model(X, Y, semipar_params, semipar_noise, model(semipar_mean_params), kernel)
                (use_model == :semiparam) && (parameters = semipar_mean_params)

            elseif param_fit_alg == :BI
                semipar_mean_param_samples, semipar_param_samples, semipar_noise_samples = sample_semipar_params(X, Y, model, gp_params_priors, noise_priors; x_dim, y_dim, kernel, mc_settings, parallel)
                semipar_models = [gp_model(X, Y, [s[:,i] for s in semipar_param_samples], [s[i] for s in semipar_noise_samples], model(semipar_mean_param_samples[:,i]), kernel) for i in 1:sample_count(mc_settings)]
                semiparametric = x -> (mapreduce(m -> m(x), .+, semipar_models) ./ length(semipar_models))  # for plotting only
                (use_model == :semiparam) && (parameters = mean(eachcol(semipar_mean_param_samples)))
            end
        else
            semiparametric = nothing
        end

        # NONPARAMETRIC MODEL (GP)
        if (make_plots && plot_all_models) || (use_model == :nonparam)
            if param_fit_alg == :MLE
                nonpar_params, nonpar_noise = opt_gps_params(X, Y, gp_params_priors, noise_priors, nothing, kernel; y_dim, multistart=param_opt_multistart, parallel, info, debug)
                nonparametric = gp_model(X, Y, nonpar_params, nonpar_noise, nothing, kernel)
            
            elseif param_fit_alg == :BI
                nonpar_param_samples, nonpar_noise_samples = sample_gps_params(X, Y, gp_params_priors, noise_priors, nothing, kernel; x_dim, mc_settings, parallel)
                nonpar_models = [gp_model(X, Y, [s[:,i] for s in nonpar_param_samples], [s[i] for s in nonpar_noise_samples], nothing, kernel) for i in 1:sample_count(mc_settings)]
                nonparametric = x -> (mapreduce(m -> m(x), .+, nonpar_models) ./ length(nonpar_models))  # for plotting only
            end
        else
            nonparametric = nothing
        end

        if !isnothing(parameters)
            info && print("  infered params: $parameters\n")
            push!(model_params_history, parameters)
        end

        # - - - - - - - - UTILITY MAXIMIZATION - - - - - - - - - - - - - - - -
        info && print("  optimizing utility ...\n")
        if param_fit_alg == :MLE
            ϵ_samples = rand(Distributions.Normal(), (y_dim, ϵ_sample_count))
        elseif param_fit_alg == :BI
            ϵ_samples = rand(Distributions.Normal(), (y_dim, sample_count(mc_settings)))
        end

        # parametric
        res_par = nothing
        if (make_plots && plot_all_models) || (use_model == :param)
            if param_fit_alg == :MLE
                acq_par = construct_acq(fitness, parametric, constraints, ϵ_samples, last(bsf))
            elseif param_fit_alg == :BI
                acqs_par_ = construct_acq.(Ref(fitness), par_models, Ref(constraints), eachcol(ϵ_samples), last(bsf))
                acq_par = x -> (mapreduce(a -> a(x), +, acqs_par_) / length(acqs_par_))
            end
            if acq_opt_alg == :CMAES
                res_par = opt_acq_CMAES(acq_par, domain; x_dim, multistart=acq_opt_multistart, discrete_dims, options=cmaes_options, parallel, info, debug)
            elseif acq_opt_alg == :Optim
                res_par = opt_acq_Optim(acq_par, domain; x_dim, multistart=acq_opt_multistart, discrete_dims, options=optim_options, parallel, info, debug)
            end
        else
            acq_par, res_par = nothing, nothing
        end

        # semiparametric
        res_semipar = nothing
        if (make_plots && plot_all_models) || (use_model == :semiparam)
            if param_fit_alg == :MLE
                acq_semipar = construct_acq(fitness, semiparametric, constraints, ϵ_samples, last(bsf))
            elseif param_fit_alg == :BI
                acqs_semipar_ = construct_acq.(Ref(fitness), semipar_models, Ref(constraints), eachcol(ϵ_samples), last(bsf))
                acq_semipar = x -> (mapreduce(a -> a(x), +, acqs_semipar_) / length(acqs_semipar_))
            end
            if acq_opt_alg == :CMAES
                res_semipar = opt_acq_CMAES(acq_semipar, domain; x_dim, multistart=acq_opt_multistart, discrete_dims, options=cmaes_options, parallel, info, debug)
            elseif acq_opt_alg == :Optim
                res_semipar = opt_acq_Optim(acq_semipar, domain; x_dim, multistart=acq_opt_multistart, discrete_dims, options=optim_options, parallel, info, debug)
            end
        else
            acq_semipar, res_semipar = nothing, nothing
        end

        # nonparametric
        res_nonpar = nothing
        if (make_plots && plot_all_models) || (use_model == :nonparam)
            if param_fit_alg == :MLE
                acq_nonpar = construct_acq(fitness, nonparametric, constraints, ϵ_samples, last(bsf))
            elseif param_fit_alg == :BI
                acqs_nonpar_ = construct_acq.(Ref(fitness), nonpar_models, Ref(constraints), eachcol(ϵ_samples), last(bsf))
                acq_nonpar = x -> (mapreduce(a -> a(x), +, acqs_nonpar_) / length(acqs_nonpar_))
            end
            if acq_opt_alg == :CMAES
                res_nonpar = opt_acq_CMAES(acq_nonpar, domain; x_dim, multistart=acq_opt_multistart, discrete_dims, options=cmaes_options, parallel, info, debug)
            elseif acq_opt_alg == :Optim
                res_nonpar = opt_acq_Optim(acq_nonpar, domain; x_dim, multistart=acq_opt_multistart, discrete_dims, options=optim_options, parallel, info, debug)
            end
        else
            acq_nonpar, res_nonpar = nothing, nothing
        end

        opt_new_x, err = select_opt_x_and_calculate_model_error(use_model, res_par, res_semipar, res_nonpar, parametric, semiparametric, nonparametric, !isnothing(errs), test_X, test_Y; info)
        isnothing(errs) || push!(errs, err)

        # - - - - - - - - PLOTTING - - - - - - - - - - - - - - - -
        if make_plots
            info && print("  plotting ...\n")
            (x_dim > 1) && throw(ErrorException("Plotting only supported for 1->N dimensional models."))

            if param_fit_alg == :BI
                if use_model == :param
                    model_samples = par_models
                    samples_lable = "param samples"
                elseif use_model == :semiparam
                    model_samples = semipar_models
                    samples_lable = "semipar samples"
                elseif use_model == :nonparam
                    model_samples = nonpar_models
                    samples_lable = "nonpar samples"
                else
                    throw(ArgumentError("Unsupported model type."))
                end
            else
                model_samples = nothing
                samples_lable = nothing
            end

            ps = create_plots(
                f_true,
                [acq_par, acq_semipar, acq_nonpar],
                [res_par, res_semipar, res_nonpar],
                [parametric, semiparametric, nonparametric],
                model_samples,
                constraints,
                X, Y;
                iter,
                domain,
                init_data_size,
                show_plots,
                param_fit_alg,
                samples_lable,
                kwargs...
            )
            push!(plots, ps)
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

    return X, Y, bsf, model_params_history, errs, plots
end

function init_Φs(lift, X)
    d = lift.(eachcol(X))
    Φs = [reduce(hcat, [ϕs[i] for ϕs in d]) for i in 1:length(d[1])]
    return Φs
end

function augment_data!(opt_new_x, fg, model::ParamModel, fitness::Fitness, domain, constraints, X, Y, Φs, F, bsf; y_dim, info)
    x_ = opt_new_x
    y_ = fg(x_)
    f_ = fitness(y_)

    info && print("  new data-point: x = $x_\n"
                * "                  y = $y_\n"
                * "                  f = $f_\n")
    X = hcat(X, x_)
    Y = hcat(Y, y_)
    push!(F, f_)

    # TODO
    # if model isa LinModel
    #     ϕs = model.lift(x_)
    #     for i in 1:y_dim
    #         Φs[i] = vcat(Φs[i], ϕs[i]')
    #     end
    # end

    # TODO: The check below caused issues together with rounding with DiscreteKernel.
    # in_domain_ = in_domain(x_, domain)

    if is_feasible(y_, constraints) && (isnothing(last(bsf)) || (f_ > last(bsf)))  # && in_domain_
        push!(bsf, f_)
    else
        push!(bsf, last(bsf))
    end

    return X, Y, Φs, F, bsf
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

function get_best_yet(F, X, Y, domain, constraints)
    in_domain = get_in_domain(X, domain)
    feasible = get_feasible(Y, constraints)
    good = in_domain .& feasible
    any(good) || return nothing
    maximum([F[i] for i in eachindex(F) if good[i]])
end

get_in_domain(X::AbstractMatrix, domain) = in_domain.(eachcol(X), Ref(domain))

get_feasible(Y::AbstractMatrix, constraints::Nothing) = fill(true, size(Y)[2])
get_feasible(Y::AbstractMatrix, constraints) = is_feasible.(eachcol(Y), Ref(constraints))

is_feasible(y::AbstractVector, constraints::Nothing) = true
is_feasible(y::AbstractVector, constraints) = all(y .< constraints)

function loglike(model, X, Y)
    ll(x, y) = logpdf(MvNormal(model(x)...), y)
    mapreduce(d -> ll(d...), +, zip(eachcol(X), eachcol(Y)))
end

end  # module
