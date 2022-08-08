using Plots
using LinearAlgebra
using Optim
using Distributions
using Turing
using Stheno
using FLoops

include("model.jl")
include("plotting.jl")
include("utils.jl")

const ModelPost = Tuple{Union{Function, Nothing}, Union{Function, Nothing}}

# TODO docs, comments & example usage
# TODO noise and hyperparams inference
# TODO refactor model error computation

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
    noise,
    constraint_noise,
    sample_count=4000,
    util_opt_multistart=200,
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

        if lin_model && lin_fitness
            ϵ_samples = nothing
        else
            ϵ_samples = rand(Distributions.Normal(), (sample_count, y_dim))  # TODO refactor: this assumes Gaussian noise
        end

        # parametric model
        if plot_all_models || (use_model == :param) || (use_model == :semiparam)
            if lin_model
                param_posts = [lin_param_posterior(Φs[i], Y[:,i], model.param_priors[i], noise[i]) for i in 1:y_dim]
                parametric = (x->lin_model_predict(x, model.lift, getindex.(param_posts, 1)), x->lin_model_vars(x, model.lift, getindex.(param_posts, 2), noise))
            else
                param_samples = sample_posterior(model.prob_model, noise, X, Y; param_count=model.param_count, sample_count)
                parametric = (x->model_predict_MC(x, model.predict, noise, param_samples, ϵ_samples; sample_count), nothing)
            end
        else
            parametric = (nothing, nothing)
        end

        # semiparametric model (param + GP)
        if plot_all_models || (use_model == :semiparam)
            semi_gps_ = gp_posterior(X, Y, noise; mean=x->parametric[1](x), kernel, y_dim)
            semiparametric = gps_pred_distr(semi_gps_)
        else
            semiparametric = (nothing, nothing)
        end

        # nonparametric model (GP)
        if plot_all_models || (use_model == :nonparam)
            pure_gps_ = gp_posterior(X, Y, noise; kernel, y_dim)
            nonparametric = gps_pred_distr(pure_gps_)
        else
            nonparametric = (nothing, nothing)
        end

        # constraint models (GPs)
        if constrained
            # TODO modify the prior mean ?
            # TODO provide option for defining semiparametric models for constraints ?
            c_gps_ = gp_posterior(X, Z, constraint_noise; kernel=constraint_kernel, y_dim=c_count)
            c_model = gps_pred_distr(c_gps_)
        else
            c_model = nothing
        end

        # - - - - - - - - UTILITY MAXIMIZATION - - - - - - - - - - - - - - - -
        info && print("  optimizing utility ...\n")
        multistart = util_opt_multistart
        res_param = res_semiparam = res_nonparam = nothing
        acq_param = acq_semiparam = acq_nonparam = nothing
        
        # parametric
        if plot_all_models || (use_model == :param)
            if lin_model
                ei_param_ = x->EI_gauss(x, fitness, parametric, ϵ_samples; best_yet=last(bsf), sample_count)
            else
                ei_param_ = x->EI_nongauss(x, fitness, model.predict, noise, param_samples, ϵ_samples; best_yet=last(bsf), sample_count)
            end
            acq_param = construct_acq_func(ei_param_, c_model; constrained, best_yet=last(bsf))
            res_param = opt_acquisition(acq_param, domain_lb, domain_ub; multistart, info, debug)
        end

        # semiparametric
        if plot_all_models || (use_model == :semiparam)
            ei_semiparam_ = x->EI_gauss(x, fitness, semiparametric, ϵ_samples; best_yet=last(bsf), sample_count)
            acq_semiparam = construct_acq_func(ei_semiparam_, c_model; constrained, best_yet=last(bsf))
            res_semiparam = opt_acquisition(acq_semiparam, domain_lb, domain_ub; multistart, info, debug)
        end

        # nonparametric
        if plot_all_models || (use_model == :nonparam)
            ei_nonparam_ = x->EI_gauss(x, fitness, nonparametric, ϵ_samples; best_yet=last(bsf), sample_count)
            acq_nonparam = construct_acq_func(ei_nonparam_, c_model; constrained, best_yet=last(bsf))
            res_nonparam = opt_acquisition(acq_nonparam, domain_lb, domain_ub; multistart, info, debug)
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
function sample_posterior(prob_model, noise, X, Y; param_count, sample_count)
    post_ = Turing.sample(prob_model(X, Y, noise), NUTS(), sample_count; verbose=false)
    return collect(eachrow(post_.value.data[:,1:param_count,1]))
end

function construct_acq_func(ei, c_model; constrained, best_yet)
    if constrained
        if isnothing(best_yet)
            return x -> constraint_weighted_acq(1., x, c_model)
        else
            return x -> constraint_weighted_acq(ei(x), x, c_model)
        end
    else
        return ei
    end
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

function opt_acquisition(acq, domain_lb, domain_ub; multistart=1, info=true, debug=false)
    dim = length(domain_lb)
    starts = rand(dim, multistart) .* (domain_ub .- domain_lb) .+ domain_lb

    results = Vector{Tuple{Vector{Float64}, Float64}}(undef, multistart)
    convergence_errors = 0
    @floop for i in 1:multistart
        try
            opt_res = Optim.optimize(x -> -acq(x), domain_lb, domain_ub, starts[:,i], Fminbox(LBFGS()))
            res = Optim.minimizer(opt_res), -Optim.minimum(opt_res)
            in_domain(res[1], domain_lb, domain_ub) || throw(ErrorException("Optimization result out of the domain."))
            results[i] = res
        catch e
            debug && throw(e)
            @reduce convergence_errors += 1
            results[i] = ([], -Inf)
        end
    end

    info && (convergence_errors > 0) && print("      $(convergence_errors.x)/$(multistart) optimization runs failed to converge!\n")
    opt_i = argmax([res[2] for res in results])
    return results[opt_i]
end

function in_domain(x, domain_lb, domain_ub)
    any(x .< domain_lb) && return false
    any(x .> domain_ub) && return false
    return true
end

function lin_model_predict(x, lift, μθs)
    return transpose.(μθs) .* lift(x)
end

function lin_model_vars(x, lift, Σθs, noise)
    ϕs = lift(x)
    return Diagonal(noise .+ (transpose.(ϕs) .* Σθs .* ϕs))
end

function model_predict_MC(x, model_predict, noise, param_samples, ϵ_samples; sample_count)
    # TODO include the noise ?
    # '+ (noise * ϵ_samples)' excluded -- The mean of ϵ should approach zero so it is unnecessary.
    return sum([model_predict(x, param_samples[i]) for i in 1:sample_count]) ./ sample_count
end

function lin_param_posterior(Φ, Y_col, param_prior, noise)
    ω = 1 / noise
    μθ, Σθ = param_prior
    inv_Σθ = inv(Σθ)

    Σθ_ = inv(inv_Σθ + ω * Φ' * Φ)
    μθ_ = Σθ_ * (inv_Σθ * μθ + ω * Φ' * Y_col)

    return μθ_, Σθ_
end

function gp_posterior(X, Y, noise; mean=nothing, kernel, y_dim=nothing)
    isnothing(y_dim) && (y_dim = size(Y)[2])
    isnothing(mean) && (mean = x->zeros(y_dim))
    return [posterior(GP(x->mean(x)[i], kernel)(X', noise[i]), Y[:,i]) for i in 1:y_dim]
end

function gps_pred_distr(gps)
    μy(x) = [mean(gp([x]))[1] for gp in gps]
    Σy(x) = Diagonal([var(gp([x]))[1] for gp in gps])
    return μy, Σy
end

# Used when: model posterior predictive distribution is Gaussian
#            and fitness is linear
function EI_gauss(x, fitness::LinFitness, model, ϵ_samples; best_yet, sample_count)
    μy, Σy = model[1](x), model[2](x)
    isdiag(Σy) || throw(ArgumentError("Model predictive distribution covariance matrix Σy must be diagonal."))

    μf = fitness.coefs' * μy
    σf = sqrt((fitness.coefs .^ 2)' * (Σy.diag .^ 2))
    
    norm_ϵ = (μf - best_yet) / σf
    return (μf - best_yet) * cdf(Distributions.Normal(), norm_ϵ) + σf * pdf(Distributions.Normal(), norm_ϵ)
end

# Used when: model posterior predictive distribution is Gaussian
#            but fitness is NOT linear
function EI_gauss(x, fitness::NonlinFitness, model, ϵ_samples; best_yet, sample_count)
    μy, Σy = model[1](x), model[2](x)
    pred_samples = [μy .+ (Σy .* ϵ_samples[i,:]) for i in 1:sample_count]
    return EI_MC(fitness, pred_samples; sample_count, best_yet)
end

# Used when: model posterior predictive distribution is NOT Gaussian
function EI_nongauss(x, fitness::Fitness, model_predict, noise, param_samples, ϵ_samples; best_yet, sample_count)
    # TODO use samples of noise instead of its mean ?
    pred_samples = [model_predict(x, param_samples[i]) .+ (noise .* ϵ_samples[i,:]) for i in 1:sample_count]
    return EI_MC(fitness, pred_samples; sample_count, best_yet)
end

function EI_MC(fitness::Fitness, pred_samples; sample_count, best_yet)
    return sum(max.(0, fitness.(pred_samples) .- best_yet)) / sample_count
end

function constraint_weighted_acq(acq, x, c_model)
    return prod(constraint_probabilities(c_model)(x)) * acq
end

function constraint_probabilities(c_model)
    function p(x)
        μ, Σ = c_model[1](x), c_model[2](x)
        isdiag(Σ) || throw(ArgumentError("The constraint GP covariance matrix is not diagonal."))
        N = length(μ)
        distrs = [Distributions.Normal(μ[i], Σ.diag[i]) for i in 1:N]
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
