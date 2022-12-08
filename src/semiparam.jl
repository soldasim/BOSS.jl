using Turing

function opt_semipar_params(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, par_model::ParamModel, gp_params_priors, noise_priors; kernel::Kernel, multistart::Int, optim_options=Optim.Options(), min_gp_hyperparam_value=1e-6, parallel, info=true, debug=false)
    x_dim = size(X)[1]
    y_dim = size(Y)[1]
    
    softplus(x) = log(1. + exp(x))
    lift(p) = softplus.(p) .+ min_gp_hyperparam_value  # 'min_gp_hyperparam_value' for numerical stability
    
    noise_loglike = noise -> mapreduce(n -> logpdf(n...), +, zip(noise_priors, noise))
    param_loglike = model_params_loglike(X, Y, par_model)
    gp_loglikes = [(params, noise, mean) -> gp_params_loglike(X, Y[i,:], gp_params_priors[i], mean, kernel)(params, noise) for i in 1:y_dim]

    split = split_opt_params_(x_dim, y_dim, par_model)

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
    constraints = (
        vcat(minimum.(noise_priors), minimum.(par_model.param_priors), vcat(minimum.(gp_params_priors)...)),
        vcat(maximum.(noise_priors), maximum.(par_model.param_priors), vcat(maximum.(gp_params_priors)...))
    )

    p, _ = optim_Optim_multistart(loglike, starts; parallel, options=optim_options, info, debug)
    noise, model_params, gp_hyperparams = split(p)
    gp_hyperparams = lift.(gp_hyperparams)
    return model_params, gp_hyperparams, noise
end

function split_opt_params_(x_dim::Int, y_dim::Int, par_model::ParamModel)
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

Turing.@model function semipar_turing_model(X::AbstractMatrix{<:Real}, Yt::AbstractMatrix{<:Real}, par_model::ParamModel, gp_params_priors, noise_priors, kernel::Kernel, y_dim::Int)
    noise ~ arraydist(noise_priors)
    model_params ~ arraydist(par_model.param_priors)
    gp_hyperparams ~ arraydist(gp_params_priors)

    mean = par_model(model_params)
    gps = [construct_finite_gp(X, gp_hyperparams[:,i], noise[i], x->mean(x)[i], kernel) for i in 1:y_dim]

    Yt ~ arraydist(gps)
end

function sample_semipar_params(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, par_model::ParamModel, gp_params_priors, noise_priors; kernel::Kernel, mc_settings::MCSettings, parallel)
    x_dim = size(X)[1]
    y_dim = size(Y)[1]
    
    model = semipar_turing_model(X, Y', par_model, gp_params_priors, noise_priors, kernel, y_dim)
    param_symbols = vcat([Symbol("noise[$i]") for i in 1:y_dim],
                         [Symbol("model_params[$i]") for i in 1:par_model.param_count],
                         reduce(vcat, [[Symbol("gp_hyperparams[$j,$i]") for j in 1:gp_param_count(x_dim)] for i in 1:y_dim]))
    samples = sample_params_turing(model, param_symbols, mc_settings; parallel)

    noise_samples, model_params_samples, gp_hyperparams_samples = split_sample_params_(x_dim, y_dim, par_model.param_count)(samples)
    return model_params_samples, gp_hyperparams_samples, noise_samples
end

function split_sample_params_(x_dim::Int, y_dim::Int, model_param_count::Int)
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

function fit_semiparametric_model(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, model::ParamModel, kernel::Kernel, gp_params_priors, noise_priors; param_fit_alg, multistart::Int=80, optim_options=Optim.Options(), mc_settings=MCSettings(400, 20, 8, 6), parallel=true, info=false, debug=false)
    if param_fit_alg == :MLE
        mean_params, gp_params, noise = opt_semipar_params(X, Y, model, gp_params_priors, noise_priors; kernel, multistart, optim_options, parallel, info, debug)
        semiparametric = gp_model(X, Y, gp_params, noise, model(mean_params), kernel)
        
        model_samples = nothing

    elseif param_fit_alg == :BI
        mean_param_samples, gp_param_samples, noise_samples = sample_semipar_params(X, Y, model, gp_params_priors, noise_priors; kernel, mc_settings, parallel)
        model_samples = [gp_model(X, Y, [s[:,i] for s in gp_param_samples], [s[i] for s in noise_samples], model(mean_param_samples[:,i]), kernel) for i in 1:sample_count(mc_settings)]
        semiparametric = x -> (mapreduce(m -> m(x), .+, model_samples) ./ length(model_samples))  # for plotting only
        
        mean_params = mean_param_samples
        gp_params = gp_param_samples
        noise = noise_samples
    end

    return semiparametric, model_samples, mean_params, gp_params, noise
end
