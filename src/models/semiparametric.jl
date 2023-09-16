using Turing

"""
Construct a new `BOSS.Semiparametric` model by wrapping its `kernel` in `BOSS.DiscreteKernel`
to define some dimensions as discrete.
"""
make_discrete(m::Semiparametric, discrete::AbstractVector{<:Bool}) =
    Semiparametric(make_discrete(m.parametric, discrete), make_discrete(m.nonparametric, discrete))

model_posterior(model::Semiparametric, data::ExperimentDataMLE) =
    model_posterior(model, data.X, data.Y, data.θ, data.length_scales, data.noise_vars)

model_posterior(model::Semiparametric, data::ExperimentDataBI) =
    model_posterior.(Ref(model), Ref(data.X), Ref(data.Y), eachcol(data.θ), data.length_scales, eachcol(data.noise_vars))

"""
Return the posterior predictive distribution of the model.

The posterior is a function `mean, var = predict(x)`
which gives the mean and variance of the predictive distribution as a function of `x`.
"""
model_posterior(
    model::Semiparametric,
    X::AbstractMatrix{NUM},
    Y::AbstractMatrix{NUM},
    θ::AbstractVector{NUM},
    length_scales::AbstractMatrix{NUM},
    noise_vars::AbstractVector{NUM},   
) where {NUM<:Real} =
    model_posterior(add_mean(model.nonparametric, model.parametric(θ)), X, Y, length_scales, noise_vars)

"""
Return the log-likelihood of the model parameters, GP hyperparameters and the noise variance
as a function `ll = loglike(θ, length_scales, noise_vars)`.
"""
function model_loglike(model::Semiparametric, noise_var_priors, data::ExperimentData)
    params_loglike = model_params_loglike(model, data.X, data.Y)
    noise_loglike(noise_vars) = mapreduce(p -> logpdf(p...), +, zip(noise_var_priors, noise_vars))
    loglike(θ, length_scales, noise_vars) = params_loglike(θ, length_scales, noise_vars) + noise_loglike(noise_vars)
end

"""
Return the log-likelihood of the model parameters and the GP hyperparameters (without the likelihood of the noise variance)
as a function `ll = loglike(θ, length_scales, noise_vars)`.
"""
function model_params_loglike(model::Semiparametric, X::AbstractMatrix{NUM}, Y::AbstractMatrix{NUM}) where {NUM<:Real}
    parametric_loglike = model_params_loglike(model.parametric, X, Y)

    function semiparametric_loglike(θ, length_scales, noise_vars)
        nonparametric_loglike = model_params_loglike(add_mean(model.nonparametric, model.parametric(θ)), X, Y)
        
        parametric_loglike(θ, noise_vars) + nonparametric_loglike(length_scales, noise_vars)
    end
end
