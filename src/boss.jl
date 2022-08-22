module Boss

using Plots
using LinearAlgebra
using Distributions
using Turing

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
# TODO NUTS/LBFGS selection for GP hyperparam opt
# TODO constraints on input

# Bayesian optimization with a N->N dimensional semiparametric surrogate model.
# Gaussian noise only. (for now)
# Plotting only works for 1D->1D problems.
#
# Uncostrained -> boss(f, X, Y, model, domain_lb, domain_ub; kwargs...)
#              ->   f(x) = y
#
# Constrained  -> boss(f, X, Y, Z, model, domain_lb, domain_ub; kwargs...)
#              ->   f(x) = (y, z)

function boss(f, fitness, X, Y, model::ParamModel, domain_lb, domain_ub; kwargs...)
    fg(x) = f(x), Float64[]
    return boss(fg, fitness, X, Y, nothing, model, domain_lb, domain_ub; kwargs...)
end

function boss(f, fitness::Function, X, Y, Z, model::ParamModel, domain_lb, domain_ub; kwargs...)
    fit = NonlinFitness(fitness)
    return boss(f, fit, X, Y, Z, model, domain_lb, domain_ub; kwargs...)
end

function boss(fg, fitness::Fitness, X, Y, Z, model::ParamModel, domain_lb, domain_ub;
    noise_priors,
    constraint_noise_priors,
    mc_sample_count=4000,
    acq_opt_multistart=200,
    gp_params_priors=nothing,
    constraint_gp_params_priors=nothing,
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
    constraint_kernel=kernel,
    use_model=:semiparam,  # :param, :semiparam, :nonparam
    gp_hyperparam_alg=:NUTS,  # :NUTS, :LBFGS
    kwargs...
)
    # - - - - - - - - INITIALIZATION - - - - - - - - - - - - - - - -
    isnothing(max_iters) && isnothing(target_err) && throw(ArgumentError("No termination condition provided. Use kwargs 'max_iters' or 'target_err' to define a termination condition."))

    lin_model = (model isa LinModel)
    lin_fitness = (fitness isa LinFitness)
    constrained = !isnothing(Z)
    c_count = constrained ? size(Z)[2] : 0
    init_data_size = size(X)[1]
    y_dim = size(Y)[2]
    x_dim = size(X)[2]
    isnothing(gp_params_priors) && (gp_params_priors = [MvLogNormal(ones(x_dim), ones(x_dim)) for _ in 1:y_dim])
    isnothing(constraint_gp_params_priors) && (constraint_gp_params_priors = [MvLogNormal(ones(x_dim), ones(x_dim)) for _ in 1:c_count])

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
            X, Y, Z, Φs, F, bsf = augment_data!(opt_new_x, fg, model, fitness, X, Y, Z, Φs, F, bsf; constrained, y_dim, info)
        end

        # - - - - - - - - MODEL INFERENCE - - - - - - - - - - - - - - - -
        info && print("  model inference ...\n")

        # sampling
        if lin_model && lin_fitness
            ϵ_samples = nothing
        else
            ϵ_samples = rand(Distributions.Normal(), (mc_sample_count, y_dim))
        end
        noise_samples = collect(eachcol(rand(Product(noise_priors), mc_sample_count)))
        c_noise_samples = collect(eachcol(rand(Product(constraint_noise_priors), mc_sample_count)))

        # parametric model
        if plot_all_models || (use_model == :param) || (use_model == :semiparam)
            if lin_model && false  # TODO refactor 'lin_param_posterior'
                param_posts = [lin_param_posterior(Φs[i], Y[:,i], model.param_priors[i], noise_priors[i]) for i in 1:y_dim]
                parametric = (x->lin_model_predict(x, model.lift, getindex.(param_posts, 1)), x->lin_model_vars(x, model.lift, getindex.(param_posts, 2), noise_priors))
            else
                lin_param_samples = sample_posterior(model.prob_model, noise_priors, X, Y; param_count=model.param_count, sample_count=mc_sample_count)
                parametric = (x->model_predict_MC(x, model.predict, lin_param_samples, noise_samples, ϵ_samples; sample_count=mc_sample_count), nothing)
            end
        else
            parametric = (nothing, nothing)
        end

        # semiparametric model (param + GP)
        if plot_all_models || (use_model == :semiparam)
            if gp_hyperparam_alg == :LBFGS
                semi_gp_params_ = [gp_fit_params_lbfgs(X, Y[:,i]; mean=x->parametric[1](x)[i], kernel, params_prior=gp_params_priors[i], noise_prior=noise_priors[i], multistart=mc_sample_count, info, debug) for i in 1:y_dim]
                semiparametric = fit_gps(X, Y, [p[1] for p in semi_gp_params_], [p[2] for p in semi_gp_params_]; y_dim, mean=parametric[1], kernel)
            elseif gp_hyperparam_alg == :NUTS
                semi_gp_params_ = [gp_sample_params_nuts(X, Y[:,i]; mean=x->parametric[1](x)[i], kernel, params_prior=gp_params_priors[i], noise_prior=noise_priors[i], sample_count=mc_sample_count) for i in 1:y_dim]
                semiparametric = fit_gps_mc(X, Y, semi_gp_params_, noise_samples; y_dim, mean=parametric[1], kernel, sample_count=mc_sample_count)
            end
        else
            semiparametric = (nothing, nothing)
        end

        # nonparametric model (GP)
        if plot_all_models || (use_model == :nonparam)
            if gp_hyperparam_alg == :LBFGS
                non_gp_params_ = [gp_fit_params_lbfgs(X, Y[:,i]; kernel, params_prior=gp_params_priors[i], noise_prior=noise_priors[i], multistart=mc_sample_count, info, debug) for i in 1:y_dim]
                nonparametric = fit_gps(X, Y, [p[1] for p in non_gp_params_], [p[2] for p in non_gp_params_]; y_dim, kernel)
            elseif gp_hyperparam_alg == :NUTS
                non_gp_params_ = [gp_sample_params_nuts(X, Y[:,i]; kernel, params_prior=gp_params_priors[i], noise_prior=noise_priors[i], sample_count=mc_sample_count) for i in 1:y_dim]
                nonparametric = fit_gps_mc(X, Y, non_gp_params_, noise_samples; y_dim, kernel, sample_count=mc_sample_count)
            end
        else
            nonparametric = (nothing, nothing)
        end

        # constraint models (GPs)
        if constrained
            # TODO modify the prior mean ?
            # TODO provide option for defining semiparametric models for constraints ?
            if gp_hyperparam_alg == :LBFGS
                c_gp_params_ = [gp_fit_params_lbfgs(X, Z[:,i]; kernel=constraint_kernel, params_prior=constraint_gp_params_priors[i], noise_prior=constraint_noise_priors[i], multistart=mc_sample_count, info, debug) for i in 1:c_count]
                c_model = fit_gps(X, Z, [p[1] for p in c_gp_params_], [p[2] for p in c_gp_params_]; y_dim=c_count, kernel=constraint_kernel)
            elseif gp_hyperparam_alg == :NUTS
                c_gp_params_ = [gp_sample_params_nuts(X, Z[:,i]; kernel=constraint_kernel, params_prior=constraint_gp_params_priors[i], noise_prior=constraint_noise_priors[i], sample_count=mc_sample_count) for i in 1:c_count]
                c_model = fit_gps_mc(X, Z, c_gp_params_, c_noise_samples; y_dim=c_count, kernel=constraint_kernel, sample_count=mc_sample_count)
            end
        else
            c_model = nothing
        end

        # - - - - - - - - UTILITY MAXIMIZATION - - - - - - - - - - - - - - - -
        info && print("  optimizing utility ...\n")
        multistart = acq_opt_multistart
        res_param = res_semiparam = res_nonparam = nothing
        acq_param = acq_semiparam = acq_nonparam = nothing
        
        # parametric
        if plot_all_models || (use_model == :param)
            if lin_model
                ei_param_ = x->EI_gauss(x, fitness, parametric, noise_samples, ϵ_samples; best_yet=last(bsf), sample_count=mc_sample_count)
            else
                ei_param_ = x->EI_nongauss(x, fitness, model.predict, lin_param_samples, noise_samples, ϵ_samples; best_yet=last(bsf), sample_count=mc_sample_count)
            end
            acq_param = construct_acq(ei_param_, c_model; constrained, best_yet=last(bsf))
            res_param = opt_acq(acq_param, domain_lb, domain_ub; multistart, info, debug)
        end

        # semiparametric
        if plot_all_models || (use_model == :semiparam)
            ei_semiparam_ = x->EI_gauss(x, fitness, semiparametric, noise_samples, ϵ_samples; best_yet=last(bsf), sample_count=mc_sample_count)
            acq_semiparam = construct_acq(ei_semiparam_, c_model; constrained, best_yet=last(bsf))
            res_semiparam = opt_acq(acq_semiparam, domain_lb, domain_ub; multistart, info, debug)
        end

        # nonparametric
        if plot_all_models || (use_model == :nonparam)
            ei_nonparam_ = x->EI_gauss(x, fitness, nonparametric, noise_samples, ϵ_samples; best_yet=last(bsf), sample_count=mc_sample_count)
            acq_nonparam = construct_acq(ei_nonparam_, c_model; constrained, best_yet=last(bsf))
            res_nonparam = opt_acq(acq_nonparam, domain_lb, domain_ub; multistart, info, debug)
        end

        opt_new_x, err = select_opt_x_and_calculate_model_error(use_model, res_param, res_semiparam, res_nonparam, parametric, semiparametric, nonparametric, !isnothing(errs), test_X, test_Y; info)
        isnothing(errs) || push!(errs, err)

        # - - - - - - - - PLOTTING - - - - - - - - - - - - - - - -
        if make_plots
            info && print("  plotting ...\n")
            (x_dim > 1) && throw(ErrorException("Plotting only supported for 1->N dimensional models."))
            ps = create_plots(
                f_true,
                [acq_param, acq_semiparam, acq_nonparam],
                [res_param, res_semiparam, res_nonparam],
                [parametric, semiparametric, nonparametric],
                c_model,
                X, Y;
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

    return X, Y, Z, bsf, errs, plots
end

function init_Φs(lift, X)
    d = lift.(eachrow(X))
    Φs = [reduce(vcat, [ϕs[i]' for ϕs in d]) for i in 1:length(d[1])]
    return Φs
end

function augment_data!(opt_new_x, fg, model, fitness, X, Y, Z, Φs, F, bsf; constrained, y_dim, info)
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
    
    if constrained
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
function sample_posterior(prob_model, noise_priors, X, Y; param_count, sample_count)
    post_ = Turing.sample(prob_model(X, Y, noise_priors), NUTS(), sample_count; verbose=false)
    return collect(eachrow(post_.value.data[:,1:param_count,1]))
end

function fit_gps_mc(X, Y, param_samples, noise_samples; y_dim, mean=x->zeros(y_dim), kernel, sample_count)
    gp_preds = [[gp_pred_distr(X, Y[:,i]; mean=x->mean(x)[i], kernel, params=param_samples[i][s], noise=noise_samples[s][i]) for s in 1:sample_count] for i in 1:y_dim]
    gp_model = (x -> [Distributions.mean([pred[1](x) for pred in gp_preds[i]]) for i in 1:y_dim],
                x -> [Distributions.mean([pred[2](x) for pred in gp_preds[i]]) for i in 1:y_dim])
    return gp_model
end

function fit_gps(X, Y, params, noise; y_dim, mean=x->zeros(y_dim), kernel)
    gp_preds = [gp_pred_distr(X, Y[:,i]; mean=x->mean(x)[i], kernel, params=params[i], noise=noise[i]) for i in 1:y_dim]
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

function model_predict_MC(x, model_predict, param_samples, noise_samples, ϵ_samples; sample_count)
    return sum([model_predict(x, param_samples[i]) .+ (noise_samples[i] .* ϵ_samples) for i in 1:sample_count]) ./ sample_count
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

function constraint_probabilities(c_model)
    function p(x)
        μ, σ = c_model[1](x), c_model[2](x)
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
