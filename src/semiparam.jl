using Turing

include("acq.jl")
include("model.jl")
include("gp.jl")
include("utils.jl")

function opt_semipar_posterior(X, Y, par_model, gp_params_priors, noise_priors; x_dim, y_dim, kernel, multistart, min_gp_hyperparam_value=1e-6, info=true, debug=false)
    softplus(x) = log(1. + exp(x))
    lift(p) = softplus.(p) .+ min_gp_hyperparam_value  # 'min_gp_hyperparam_value' for numerical stability
    
    noise_loglike = noise -> sum([logpdf(noise_priors[i], noise[i]) for i in 1:y_dim])
    par_loglike = model_params_loglike(X, Y, par_model)
    gp_loglikes = [(params, noise, mean) -> gp_params_loglike(X, Y[:,i], gp_params_priors[i]; mean, kernel)(params, noise) for i in 1:y_dim]

    split = split_opt_params_(; x_dim, y_dim, par_model)

    function loglike(p)
        noise, model_params, gp_hyperparams = split(p)
        gp_hyperparams = lift.(gp_hyperparams)

        mean = x -> par_model.predict(x, model_params)
        return noise_loglike(noise) + par_loglike(model_params, noise) + sum([gp_loglikes[i](gp_hyperparams[i], noise[i], x->mean(x)[i]) for i in 1:y_dim])
    end

    starts = reduce(hcat, vcat(
        [rand(np, multistart) for np in noise_priors],
        [rand(pp, multistart) for pp in par_model.param_priors],
        [rand(pp, multistart)' for pp in gp_params_priors]
    ))

    p, _ = optim_params(loglike, starts; info, debug)
    noise, model_params, gp_hyperparams = split(p)
    gp_hyperparams = lift.(gp_hyperparams)
    return model_params, gp_hyperparams, noise
end

function split_opt_params_(; x_dim, y_dim, par_model)
    n_i = 1
    mp_i = n_i + y_dim
    gphp_i = mp_i + par_model.param_count
    gphp_count = gp_param_count(x_dim)

    function split(p)
        noise = p[n_i:mp_i-1]
        model_params = p[mp_i:gphp_i-1]
        gp_hyperparams = [p[gphp_i+((i-1)*gphp_count):gphp_i+(i*gphp_count)-1] for i in 1:y_dim]

        return noise, model_params, gp_hyperparams
    end
    return split
end

function sample_semipar_posterior(X, Y, par_model::ParamModel, gp_params_priors, noise_priors; x_dim, y_dim, kernel, mc_settings::MCSettings)
    Turing.@model function semipar_model(X, Y, par_model, gp_params_priors, noise_priors, kernel, x_dim, y_dim)
        noise = Vector{Float64}(undef, y_dim)
        for i in 1:y_dim
            noise[i] ~ noise_priors[i]
        end

        model_params = Vector{Float64}(undef, par_model.param_count)
        for i in 1:par_model.param_count
            model_params[i] ~ par_model.param_priors[i]
        end

        gp_hyperparams = [Vector{Float64}(undef, gp_param_count(x_dim)) for _ in 1:y_dim]
        for i in 1:y_dim
            gp_hyperparams[i] ~ gp_params_priors[i]
        end

        mean = x -> par_model.predict(x, model_params)
        gps = [construct_finite_gp(X, gp_hyperparams[i], noise[i]; mean=x->mean(x)[i], kernel) for i in 1:y_dim]
        for i in 1:y_dim
            Y[:,i] ~ gps[i]
        end
    end
    
    model = semipar_model(X, Y, par_model, gp_params_priors, noise_priors, kernel, x_dim, y_dim)
    param_symbols = vcat([Symbol("noise[$i]") for i in 1:y_dim],
                         [Symbol("model_params[$i]") for i in 1:par_model.param_count],
                         reduce(vcat, [[Symbol("gp_hyperparams[$i][$j]") for j in 1:gp_param_count(x_dim)] for i in 1:y_dim]))
    samples = sample_params_nuts(model, param_symbols, mc_settings)
    
    noise_samples, model_params_samples, gp_hyperparams_samples = split_sample_params_(; x_dim, y_dim, par_model, sample_count=sample_count(mc_settings))(samples)
    return model_params_samples, gp_hyperparams_samples, noise_samples
end

function split_sample_params_(; x_dim, y_dim, par_model, sample_count)
    n_i = 1
    mp_i = n_i + y_dim
    gphp_i = mp_i + par_model.param_count
    gphp_count = gp_param_count(x_dim)

    function split(samples)
        noise = reduce(hcat, samples[n_i:mp_i-1])
        model_params = reduce(hcat, samples[mp_i:gphp_i-1])
        gp_hyperparams = [reduce(hcat, samples[gphp_i+((i-1)*gphp_count):gphp_i+(i*gphp_count)-1]) for i in 1:y_dim]

        noise_samples = collect(eachrow(noise))
        model_params_samples = collect(eachrow(model_params))
        gp_hyperparams_samples = [[gp_hyperparams[i][s,:] for i in 1:y_dim] for s in 1:sample_count]

        return noise_samples, model_params_samples, gp_hyperparams_samples
    end

    return split
end
