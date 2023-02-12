using Turing

model_posterior(model::Semiparametric, data::ExperimentDataMLE) =
    model_posterior(model, data.X, data.Y, data.θ, data.length_scales, data.noise_vars)

model_posterior(model::Semiparametric, data::ExperimentDataBI) =
    model_posterior.(Ref(model), Ref(data.X), Ref(data.Y), eachcol(data.θ), data.length_scales, eachcol(data.noise_vars))

model_posterior(
    model::Semiparametric,
    X::AbstractMatrix{NUM},
    Y::AbstractMatrix{NUM},
    θ::AbstractArray{NUM},
    length_scales::AbstractMatrix{NUM},
    noise_vars::AbstractArray{NUM},   
) where {NUM<:Real} =
    model_posterior(add_mean(model.nonparametric, model.parametric(θ)), X, Y, length_scales, noise_vars)

# Log-likelihood of model parameters, hyperparameters and noise variance.
function model_loglike(model::Semiparametric, noise_var_priors, data::ExperimentData)
    params_loglike = model_params_loglike(model, data.X, data.Y)
    noise_loglike(noise_vars) = mapreduce(p -> logpdf(p...), +, zip(noise_var_priors, noise_vars))
    loglike(θ, length_scales, noise_vars) = params_loglike(θ, length_scales, noise_vars) + noise_loglike(noise_vars)
end

# Log-likelihood of model parameters and hyperparameters.
function model_params_loglike(model::Semiparametric, X::AbstractMatrix{NUM}, Y::AbstractMatrix{NUM}) where {NUM<:Real}
    parametric_loglike = model_params_loglike(model.parametric, X, Y)

    function semiparametric_loglike(θ, length_scales, noise_vars)
        nonparametric_loglike = model_params_loglike(add_mean(model.nonparametric, model.parametric(θ)), X, Y)
        
        parametric_loglike(θ, noise_vars) + nonparametric_loglike(length_scales, noise_vars)
    end
end

add_mean(n::Nonparametric{Nothing}, mean::Base.Callable) = Nonparametric(mean, n.kernel, n.length_scale_priors)
