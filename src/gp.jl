using AbstractGPs
using FLoops

MIN_NOISE() = 1e-6

function gp_pred_distr(X, y; mean=x->0., kernel, params, noise)
    post_gp = gp_posterior(X, y; mean, kernel, params, noise)
    return gp_pred_distr(post_gp)
end
function gp_pred_distr(posterior_gp)
    μ(x) = mean(posterior_gp([x]))[1]
    σ(x) = var(posterior_gp([x]))[1]
    return μ, σ
end

function gp_posterior(X, y; mean=x->0., kernel, params, noise)
    gp = construct_finite_gp(X; mean, kernel, params, noise)
    return gp_posterior(gp, y)
end
function gp_posterior(finite_gp, y)
    return posterior(finite_gp, y)
end

function gp_param_len(x_dim)
    return x_dim
end
function construct_finite_gp(X; mean=x->0., kernel, params, noise)
    all(params .> 0.) || throw(ArgumentError("Params must be positive."))
    (noise < 0.) && throw(ArgumentError("Noise cannot be negative."))

    # kernel = kernel ∘ ScaleTransform(params[1])
    kernel = with_lengthscale(kernel, params)  # TODO add amplitude? ( * amp)  "https://discourse.julialang.org/t/gaussian-process-model-with-turing/42453/13"
    return GP(mean, kernel)(X', noise + MIN_NOISE())  # 'MIN_NOISE' for numerical stability
end

function gp_sample_params_nuts(X, y; mean=x->0., kernel, params_prior, noise_prior, sample_count)
    Turing.@model function gp_model(X, y, mean, kernel, params_prior, noise_prior)
        params ~ params_prior
        noise ~ noise_prior
        gp = construct_finite_gp(X; mean, kernel, params, noise)
        y ~ gp
    end

    chain = Turing.sample(gp_model(X, y, mean, kernel, params_prior, noise_prior), NUTS(), sample_count; verbose=false)
    param_samples = collect(eachrow(chain.value.data[:,1:gp_param_len(size(X)[2]),1]))
    # return Distributions.mean(param_samples)
    return param_samples
end

function gp_params_loglikelihood(X, y; mean=x->0., kernel, params_prior, noise_prior)
    function logposterior(params, noise)
        gp = construct_finite_gp(X; mean, kernel, params, noise)
        return logpdf(gp, y)
    end
    
    function loglike(p)
        noise, params... = p
        return logposterior(params, noise) + logpdf(params_prior, params) + logpdf(noise_prior, noise)
    end

    return loglike
end

function gp_fit_params_lbfgs(X, y; mean=x->0., kernel, params_prior, noise_prior, multistart=1, info=true, debug=false)
    loglike = gp_params_loglikelihood(X, y; mean, kernel, params_prior, noise_prior)
    opt_params = gp_fit_params_lbfgs(loglike; multistart, params_prior, noise_prior, info, debug)
    return opt_params
end
function gp_fit_params_lbfgs(loglike; multistart=1, params_prior, noise_prior, info=true, debug=false, min_param_value=0.0001)
    softplus(x) = log(1. + exp(x))
    lift(p) = softplus.(p) .+ min_param_value
    starts = hcat(rand(noise_prior, multistart), rand(params_prior, multistart)')

    results = Vector{Tuple{Vector{Float64}, Float64}}(undef, multistart)
    convergence_errors = 0
    @floop for i in 1:multistart
        try
            opt_res = Optim.optimize(p -> -loglike(lift(p)), starts[i,:], LBFGS())
            results[i] = lift(Optim.minimizer(opt_res)), -Optim.minimum(opt_res)
        catch e
            debug && throw(e)
            @reduce convergence_errors += 1
            results[i] = ([], -Inf)
        end
    end

    info && (convergence_errors > 0) && print("      $(convergence_errors)/$(multistart) optimization runs failed to converge!\n")
    opt_i = argmax([res[2] for res in results])
    noise, params... = results[opt_i][1]
    return params, noise
end
