using AbstractGPs
using FLoops

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

function gp_param_len(x_dim)
    return x_dim
end
function construct_finite_gp(X, params, noise; mean=x->0., kernel, min_noise=1e-6)
    all(params .> 0.) || throw(ArgumentError("Params must be positive."))
    (noise < 0.) && throw(ArgumentError("Noise cannot be negative."))

    kernel = with_lengthscale(kernel, params)  # TODO add amplitude? ( * amp)  "https://discourse.julialang.org/t/gaussian-process-model-with-turing/42453/13"
    return GP(mean, kernel)(X', max(noise, min_noise))  # 'min_noise' for numerical stability
end

function gp_sample_params_nuts(X, y, params_prior, noise_prior; x_dim, mean=x->0., kernel, sample_count)
    Turing.@model function gp_model(X, y, mean, kernel, params_prior, noise_prior)
        params ~ params_prior
        noise ~ noise_prior
        gp = construct_finite_gp(X, params, noise; mean, kernel)
        y ~ gp
    end

    chain = Turing.sample(gp_model(X, y, mean, kernel, params_prior, noise_prior), NUTS(), sample_count; verbose=false)
    params = reduce(hcat, [chain[Symbol("params[$i]")][:] for i in 1:gp_param_len(x_dim)])
    noise = chain[:noise][:]
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

function gp_fit_params_lbfgs(X, y, params_prior, noise_prior; mean=x->0., kernel, multistart, info=true, debug=false)
    loglike = gp_params_loglikelihood(X, y, params_prior, noise_prior; mean, kernel)
    opt_params = gp_fit_params_lbfgs(loglike, params_prior, noise_prior; multistart, info, debug)
    return opt_params
end
function gp_fit_params_lbfgs(loglike, params_prior, noise_prior; multistart, info=true, debug=false, min_param_value=1e-6)
    softplus(x) = log(1. + exp(x))
    lift(p) = softplus.(p) .+ min_param_value  # 'min_param_value' for numerical stability
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
