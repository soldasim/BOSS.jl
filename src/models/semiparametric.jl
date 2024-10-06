
"""
    Nonparametric(; kwargs...)

Alias for [`GaussianProcess`](@ref).
"""
const Nonparametric = GaussianProcess

"""
    Semiparametric(; kwargs...)

A semiparametric surrogate model (a combination of a parametric model and Gaussian Process).

The parametric model is used as the mean of the Gaussian Process and all parameters
and hyperparameters are estimated simultaneously.

# Keywords
- `parametric::Parametric`: The parametric model used as the GP mean function.
- `nonparametric::Nonparametric{Nothing}`: The outer GP model without mean.

Note that the parametric model must be defined without noise priors,
and the nonparametric model must be defined without mean function.
"""
struct Semiparametric{
    P<:Parametric{Nothing},     # parametric model without noise std priors
    N<:Nonparametric{Nothing},  # nonparametric model without mean function
} <: SurrogateModel
    parametric::P
    nonparametric::N
end
Semiparametric(;
    parametric,
    nonparametric,
) = Semiparametric(
    remove_noise_priors(parametric),
    remove_mean(nonparametric),
)

make_discrete(m::Semiparametric, discrete::AbstractVector{<:Bool}) =
    Semiparametric(make_discrete(m.parametric, discrete), make_discrete(m.nonparametric, discrete))

model_posterior(model::Semiparametric, data::ExperimentDataMAP) =
    model_posterior(add_mean(model.nonparametric, model.parametric(data.params[1])), data)

model_posterior_slice(model::Semiparametric, data::ExperimentDataMAP, slice::Int) =
    model_posterior_slice(add_mean(model.nonparametric, model.parametric(data.params[1])), data, slice)

function model_loglike(model::Semiparametric, data::ExperimentData)
    function loglike(params)
        ll_data = data_loglike(model, data.X, data.Y, params)
        ll_params = model_params_loglike(model, params)
        return ll_data + ll_params
    end
end

function data_loglike(
    model::Semiparametric,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    params::ModelParams,
)
    θ, λ, α, noise_std = params
    return data_loglike(
        add_mean(model.nonparametric, model.parametric(θ)),
        X,
        Y,
        params,
    )
end

function model_params_loglike(model::Semiparametric, params)
    θ, λ, α, noise_std = params
    ll_theta = sum(logpdf.(model.parametric.theta_priors, θ); init=0.)
    ll_λ = sum(logpdf.(model.nonparametric.length_scale_priors, eachcol(λ)))
    ll_α = sum(logpdf.(model.nonparametric.amp_priors, α))
    ll_noise = sum(logpdf.(model.nonparametric.noise_std_priors, noise_std))
    return ll_theta + ll_λ + ll_α + ll_noise
end

function sample_params(model::Semiparametric)
    θ = isempty(model.parametric.theta_priors) ?
            Float64[] :
            rand.(model.parametric.theta_priors)
    λ = reduce(hcat, rand.(model.nonparametric.length_scale_priors))
    α = rand.(model.nonparametric.amp_priors)
    noise_std = rand.(model.nonparametric.noise_std_priors)
    return θ, λ, α, noise_std
end

function param_priors(model::Semiparametric)
    θ_priors = model.parametric.theta_priors
    λ_priors = model.nonparametric.length_scale_priors
    α_priors = model.nonparametric.amp_priors
    noise_std_priors = model.nonparametric.noise_std_priors
    return θ_priors, λ_priors, α_priors, noise_std_priors
end
