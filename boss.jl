using Plots
using LinearAlgebra
using Optim
using Distributions
using Turing
using Stheno

include("model.jl")
include("plotting.jl")
include("utils.jl")

const ModelPost = Tuple{Union{Function, Nothing}, Union{Function, Nothing}}

# TODO docs & example usage
# TODO noise and hyperparams inference

# Bayesian optimization with a Nx1 dimensional parametric model.
# Assumes Gaussian noise! (for now)
# Plotting only works for 1D->1D problems.
#
# Uncostrained -> boss(f, X, Y, model, domain_lb, domain_ub; kwargs...)
#              ->   f(x) = y
#
# Constrained  -> boss(f, X, Y, Z, model, domain_lb, domain_ub; kwargs...)
#              ->   f(x) = (y, g1, g2, ...)

function boss(f, fitness, X, Y, model, domain_lb, domain_ub; kwargs...)
    fg(x) = f(x), Float64[]
    return boss(fg, fitness, X, Y, nothing, model, domain_lb, domain_ub; kwargs...)
end

function boss(fg, fitness, X, Y, Z, model, domain_lb, domain_ub;
    noise,
    constraint_noise,
    sample_count=200,
    util_opt_multistart=100,
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
    isnothing(max_iters) && isnothing(target_err) && throw(ArgumentError("No termination condition provided. Use kwargs 'max_iters' or 'target_err' to define a termination condition."))

    constrained = !isnothing(Z)
    c_count = constrained ? size(Z)[2] : 0
    init_data_size = size(X)[1]
    y_dim = size(Y)[2]
    x_dim = size(X)[2]

    F = [fitness(y) for y in eachrow(Y)]
    bsf = [get_best_yet(F, Z; data_size=init_data_size)]

    plots = make_plots ? Plots.Plot[] : nothing
    errs = (isnothing(test_X) || isnothing(test_Y)) ? nothing : Vector{Float64}[]

    iter = 0
    opt_new_x = Float64[]
    while true
        info && print("\nITER $iter\n")

        # NEW DATA - - - - - - - - - - - - - - - -
        if iter != 0
            info && print("  evaluating the objective function ...\n")
            X, Y, Z, F, bsf = augment_data!(opt_new_x, fg, fitness, X, Y, Z, F, bsf; constrained, info)
        end

        # MODEL INFERENCE - - - - - - - - - - - - - - - -
        info && print("  model inference ...\n")

        ϵ_samples = rand(Distributions.Normal(), (sample_count, y_dim))  # TODO refactor (this assumes Gaussian noise)

        # parametric model
        if plot_all_models || (use_model == :param) || (use_model == :semiparam)
            post_ = sample_posterior(model.prob_model, noise, X, Y; sample_count)
            param_samples = post_.value.data[:,1:model.param_count,1] #post_.params
            parametric = (x -> model_predict_MC(model.predict, x; param_samples, ϵ_samples, noise, sample_count), nothing)
        else
            parametric = (nothing, nothing)
        end

        # semiparametric model (param + GP)
        if plot_all_models || (use_model == :semiparam)
            semi_gps_ = gp_posterior(X, Y; mean=x->parametric[1](x), kernel, noise, y_dim)
            semiparametric = gps_pred_distr(semi_gps_)
        else
            semiparametric = (nothing, nothing)
        end

        # nonparametric model (GP)
        if plot_all_models || (use_model == :nonparam)
            pure_gps_ = gp_posterior(X, Y; kernel, noise, y_dim)
            nonparametric = gps_pred_distr(pure_gps_)
        else
            nonparametric = (nothing, nothing)
        end

        # constraint models (GPs)
        if constrained
            # TODO modify the prior mean ?
            # TODO provide option for defining semiparametric models for constraints ?
            c_gps_ = gp_posterior(X, Z; kernel=constraint_kernel, noise=constraint_noise, y_dim=c_count)
            c_model = gps_pred_distr(c_gps_)
        else
            c_model = nothing
        end

        # UTILITY MAXIMIZATION - - - - - - - - - - - - - - - -
        info && print("  optimizing utility ...\n")
        multistart = util_opt_multistart
        res_param = res_semiparam = res_nonparam = nothing
        acq_param = acq_semiparam = acq_nonparam = nothing
        
        # parametric
        if plot_all_models || (use_model == :param)
            ei_param_ = x->EI_param(fitness, model.predict, x; best_yet=last(bsf), noise, param_samples, ϵ_samples, sample_count)
            acq_param = construct_acq_func(ei_param_, c_model; constrained, best_yet=last(bsf))
            res_param = opt_acquisition(acq_param, domain_lb, domain_ub; multistart, info, debug)
        end

        # semiparametric
        if plot_all_models || (use_model == :semiparam)
            ei_semiparam_ = x->EI_GP(fitness, semiparametric, x; best_yet=last(bsf), noise, ϵ_samples, sample_count)
            acq_semiparam = construct_acq_func(ei_semiparam_, c_model; constrained, best_yet=last(bsf))
            res_semiparam = opt_acquisition(acq_semiparam, domain_lb, domain_ub; multistart, info, debug)
        end

        # nonparametric
        if plot_all_models || (use_model == :nonparam)
            ei_nonparam_ = x->EI_GP(fitness, nonparametric, x; best_yet=last(bsf), noise, ϵ_samples, sample_count)
            acq_nonparam = construct_acq_func(ei_nonparam_, c_model; constrained, best_yet=last(bsf))
            res_nonparam = opt_acquisition(acq_nonparam, domain_lb, domain_ub; multistart, info, debug)
        end

        opt_new_x, err = select_opt_x_and_calculate_model_error(use_model, res_param, res_semiparam, res_nonparam, parametric, semiparametric, nonparametric, !isnothing(errs), test_X, test_Y; info)
        isnothing(errs) || push!(errs, err)

        # PLOTTING - - - - - - - - - - - - - - - -
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

        # TERMINATION CONDITIONS - - - - - - - -
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

function augment_data!(opt_new_x, fg, fitness, X, Y, Z, F, bsf; constrained, info)
    x_ = opt_new_x
    y_, z_ = fg(x_)
    f_ = fitness(y_)

    info && print("  new data-point: x = $x_\n"
                * "                  y = $y_\n"
                * "                  f = $f_\n")
    X = vcat(X, x_')
    Y = vcat(Y, [y_...]')
    F = vcat(F, f_)
    
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

    return X, Y, Z, F, bsf
end

# Sample from the posterior parameter distributions given the data 'X', 'Y'.
function sample_posterior(prob_model, noise, X, Y; sample_count=4000)
    return sample(prob_model(X, Y, noise), NUTS(), sample_count;);
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

function model_predict_MC(model_predict, x; param_samples, ϵ_samples, noise, sample_count)
    # TODO include the noise ?
    # '+ (noise * ϵ_samples)' excluded -- The mean of ϵ should approach zero so it is unnecessary.
    return sum([model_predict(x, param_samples[i,:]) for i in 1:sample_count]) ./ sample_count
end

function opt_acquisition(acq, domain_lb, domain_ub; multistart=1, info=true, debug=false)
    dim = length(domain_lb)
    starts = rand(dim, multistart) .* (domain_ub .- domain_lb) .+ domain_lb

    best_res = (Float64[], -Inf)
    convergence_errors = 0
    for i in 1:multistart
        try
            opt_res = Optim.optimize(x -> -acq(x), domain_lb, domain_ub, starts[:,i], Fminbox(NelderMead()))
            res = Optim.minimizer(opt_res), -Optim.minimum(opt_res)
            (res[2] > best_res[2]) && (best_res = res) 
        catch e
            convergence_errors += 1
            debug && throw(e)
        end
    end

    info && (convergence_errors > 0) && print("    $convergence_errors/$multistart optimization runs failed to converge!\n")
    return best_res
end

function gp_posterior(X, Y; mean=nothing, kernel, noise, y_dim=nothing)
    isnothing(y_dim) && (y_dim = size(Y)[2])
    isnothing(mean) && (mean = x->zeros(y_dim))
    return [posterior(GP(x->mean(x)[i], kernel)(X', noise[i]), Y[:,i]) for i in 1:y_dim]
end

function gps_pred_distr(gps)
    μy(x) = [mean(gp([x]))[1] for gp in gps]
    σy(x) = [var(gp([x]))[1] for gp in gps]
    return μy, σy
end

function EI_param(fitness, model_predict, x; noise, param_samples, ϵ_samples, sample_count, best_yet)
    # TODO use samples of noise instead of its mean ?
    pred_samples = [model_predict(x, param_samples[i,:]) .+ (noise .* ϵ_samples[i,:]) for i in 1:sample_count]
    return EI(fitness, pred_samples; sample_count, best_yet)
end

# Analytical version, works only without 'fitness'
# function EI_GP(model, x; best_yet)
#     μy, σy = model[1](x), model[2](x)
#     norm_ϵ = (μy - best_yet) / σy
#     return (μy - best_yet) * cdf(Distributions.Normal(), norm_ϵ) + σy * pdf(Distributions.Normal(), norm_ϵ)
# end

function EI_GP(fitness, model, x; noise, ϵ_samples, sample_count, best_yet)
    # TODO implement an option for defining the fitness integral analytically to avoid sample-approximation

    # Add the predictive distribution and the noise distribution (both are normal).
    μ = model[1](x) #.+ 0.
    σ = model[2](x) .+ noise

    pred_samples = [μ .+ (σ .* ϵ_samples[i,:]) for i in 1:sample_count]
    return EI(fitness, pred_samples; sample_count, best_yet)
end

function EI(fitness, pred_samples; sample_count, best_yet)
    return sum(max.(0, fitness.(pred_samples) .- best_yet)) / sample_count
end

function constraint_weighted_acq(acq, x, c_model)
    return prod(constraint_probabilities(c_model)(x)) * acq
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

function model_dim_slice(model, dim)
    μ = isnothing(model[1]) ? nothing : x -> model[1](x)[dim]
    σ = isnothing(model[2]) ? nothing : x -> model[2](x)[dim]
    return μ, σ
end
