using AbstractGPs
using FLoops

include("utils.jl")

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

function gp_params_loglikelihood(X, y, params_prior, noise_prior; mean=x->0., kernel)
    function logposterior(params, noise)
        gp = construct_finite_gp(X, params, noise; mean, kernel)
        return logpdf(gp, y)
    end
    
    function loglike(p)
        noise, params... = p
        return logposterior(params, noise) + logpdf(params_prior, params) + logpdf(noise_prior, noise)
    end

    return loglike
end

function fit_gp_params_lbfgs(X, y, params_prior, noise_prior; mean=x->0., kernel, multistart, info=true, debug=false, min_param_value=1e-6)
    softplus(x) = log(1. + exp(x))
    lift(p) = softplus.(p) .+ min_param_value  # 'min_param_value' for numerical stability
    loglike = gp_params_loglikelihood(X, y, params_prior, noise_prior; mean, kernel)
    
    starts = hcat(rand(noise_prior, multistart), rand(params_prior, multistart)')
    p, _ = optim_params(p -> loglike(lift(p)), starts; info, debug)
    noise, params... = lift(p)
    return params, noise
end
