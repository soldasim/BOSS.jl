using AbstractGPs

include("utils.jl")

function fit_gps(X, Y, params, noise; y_dim, mean=x->zeros(y_dim), kernel)
    gp_preds = [gp_pred_distr(X, Y[:,i], params[i], noise[i]; mean=x->mean(x)[i], kernel) for i in 1:y_dim]
    return fit_gps(gp_preds)
end
function fit_gps(finite_gps, Y)
    gp_preds = gp_pred_distr.(gp_posterior.(finite_gps, collect(eachcol(Y))))
    return fit_gps(gp_preds)
end
function fit_gps(gp_preds)
    return (x -> [pred[1](x) for pred in gp_preds],
            x -> [pred[2](x) for pred in gp_preds])
end

function gp_pred_distr(X, y, params, noise; mean=x->0., kernel)
    post_gp = gp_posterior(X, y, params, noise; mean, kernel)
    return gp_pred_distr(post_gp)
end
function gp_pred_distr(posterior_gp)
    μ(x) = mean(posterior_gp([x]))[1]
    σ(x) = var(posterior_gp([x]))[1]
    return μ, σ
end

function gp_posterior(X, y, params, noise; mean=x->0., kernel)
    gp = construct_finite_gp(X, params, noise; mean, kernel)
    return gp_posterior(gp, y)
end
function gp_posterior(finite_gp, y)
    return posterior(finite_gp, y)
end

function gp_param_count(x_dim)
    return x_dim
end

function construct_finite_gps(X, params, noise; y_dim, mean=x->ones(y_dim), kernel, min_param_val=1e-6, min_noise=1e-6)
    return [construct_finite_gp(X, params[i], noise[i]; mean=x->mean(x)[i], kernel, min_param_val, min_noise) for i in 1:y_dim]
end
function construct_finite_gp(X, params, noise; mean=x->0., kernel, min_param_val=1e-6, min_noise=1e-6)
    any(params .< 0.) && throw(ArgumentError("Params must be positive. Got '$params'."))
    (noise < 0.) && throw(ArgumentError("Noise must be positive. Got '$noise'."))

    # for numerical stability
    params .+= min_param_val
    noise += min_noise

    kernel = with_lengthscale(kernel, params)
    return GP(mean, kernel)(X', noise)
end

function sample_gp_params_nuts(X, y, params_prior, noise_prior; x_dim, mean=x->0., kernel, mc_settings::MCSettings)
    Turing.@model function gp_model(X, y, mean, kernel, params_prior, noise_prior)
        params ~ params_prior
        noise ~ noise_prior
        gp = construct_finite_gp(X, params, noise; mean, kernel)
        y ~ gp
    end
    
    model = gp_model(X, y, mean, kernel, params_prior, noise_prior)
    param_symbols = vcat([Symbol("params[$i]") for i in 1:gp_param_count(x_dim)], :noise)
    samples = sample_params_nuts(model, param_symbols, mc_settings)
    params = reduce(hcat, samples[1:gp_param_count(x_dim)])
    noise = samples[end]
    return params, noise
end

function gp_params_loglike(X, y, params_prior; mean=x->0., kernel)
    function logposterior(params, noise)
        gp = construct_finite_gp(X, params, noise; mean, kernel)
        return logpdf(gp, y)
    end

    function loglike(params, noise)
        return logposterior(params, noise) + logpdf(params_prior, params)
    end
    return loglike
end

function fit_gp_params_lbfgs(X, y, params_prior, noise_prior; mean=x->0., kernel, multistart, info=true, debug=false, min_param_value=1e-6)
    softplus(x) = log(1. + exp(x))
    lift(p) = softplus.(p) .+ min_param_value  # 'min_param_value' for numerical stability
    
    params_loglike = gp_params_loglike(X, y, params_prior; mean, kernel)
    noise_loglike = noise -> logpdf(noise_prior, noise)

    function loglike(p)
        noise, params... = p
        return params_loglike(params, noise) + noise_loglike(noise)
    end

    starts = hcat(rand(noise_prior, multistart), rand(params_prior, multistart)')
    
    p, _ = optim_params(p -> loglike(lift(p)), starts; info, debug)
    noise, params... = lift(p)
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
