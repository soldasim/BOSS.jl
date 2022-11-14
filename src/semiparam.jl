using Turing

function opt_semipar_params(X, Y, par_model, gp_params_priors, noise_priors; x_dim, y_dim, kernel, multistart, min_gp_hyperparam_value=1e-6, parallel, info=true, debug=false)
    softplus(x) = log(1. + exp(x))
    lift(p) = softplus.(p) .+ min_gp_hyperparam_value  # 'min_gp_hyperparam_value' for numerical stability
    
    noise_loglike = noise -> mapreduce(n -> logpdf(n...), +, zip(noise_priors, noise))
    param_loglike = model_params_loglike(X, Y, par_model)
    gp_loglikes = [(params, noise, mean) -> gp_params_loglike(X, Y[i,:], gp_params_priors[i], mean, kernel)(params, noise) for i in 1:y_dim]

    split = split_opt_params_(; x_dim, y_dim, par_model)

    function gps_ll(gp_hyperparams, noise, mean)
        ll = 0.
        for i in 1:y_dim
            ll += gp_loglikes[i](gp_hyperparams[i], noise[i], x->mean(x)[i])
        end
        return ll
    end

    function loglike(p)
        noise, model_params, gp_hyperparams = split(p)
        gp_hyperparams = lift.(gp_hyperparams)

        noise_loglike(noise) + param_loglike(model_params, noise) + gps_ll(gp_hyperparams, noise, par_model(model_params))
    end

    starts = reduce(vcat, vcat(
        [rand(np, multistart)' for np in noise_priors],
        [rand(pp, multistart)' for pp in par_model.param_priors],
        [rand(pp, multistart) for pp in gp_params_priors],
    ))

    p, _ = optim_params(loglike, starts; parallel, info, debug)
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
end

# Needed for `arraydist` to work with multivariate distributions.
Bijectors.bijector(::Turing.DistributionsAD.VectorOfMultivariate) = Bijectors.PDBijector()

Turing.@model function semipar_turing_model(X, Y_inv, par_model, gp_params_priors, noise_priors, kernel, y_dim)
    noise ~ arraydist(noise_priors)
    model_params ~ arraydist(par_model.param_priors)
    gp_hyperparams ~ arraydist(gp_params_priors)

    mean = par_model(model_params)
    gps = [construct_finite_gp(X, gp_hyperparams[:,i], noise[i], x->mean(x)[i], kernel) for i in 1:y_dim]

    Y_inv ~ arraydist(gps)
end

function sample_semipar_params(X, Y, par_model::ParamModel, gp_params_priors, noise_priors; x_dim, y_dim, kernel, mc_settings::MCSettings, parallel)
    model = semipar_turing_model(X, Y', par_model, gp_params_priors, noise_priors, kernel, y_dim)
    param_symbols = vcat([Symbol("noise[$i]") for i in 1:y_dim],
                         [Symbol("model_params[$i]") for i in 1:par_model.param_count],
                         reduce(vcat, [[Symbol("gp_hyperparams[$j,$i]") for j in 1:gp_param_count(x_dim)] for i in 1:y_dim]))
    samples = sample_params_turing(model, param_symbols, mc_settings; parallel)

    noise_samples, model_params_samples, gp_hyperparams_samples = split_sample_params_(x_dim, y_dim, par_model.param_count, sample_count(mc_settings))(samples)
    return model_params_samples, gp_hyperparams_samples, noise_samples
end

function split_sample_params_(x_dim, y_dim, model_param_count, sample_count)
    n_i = 1
    mp_i = n_i + y_dim
    gphp_i = mp_i + model_param_count
    gphp_count = gp_param_count(x_dim)

    function split(samples)
        noise_samples = samples[n_i:mp_i-1]
        model_params_samples = reduce(vcat, transpose.(samples[mp_i:gphp_i-1]))
        gp_hyperparams_samples = [reduce(vcat, transpose.(samples[gphp_i+((i-1)*gphp_count):gphp_i+(i*gphp_count)-1])) for i in 1:y_dim]
        
        return noise_samples, model_params_samples, gp_hyperparams_samples
    end
end
