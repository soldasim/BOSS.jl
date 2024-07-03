
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
"""
struct Semiparametric{
    P<:Parametric,
    N<:Nonparametric{Nothing},
} <: SurrogateModel
    parametric::P
    nonparametric::N
end
Semiparametric(;
    parametric,
    nonparametric,
) = Semiparametric(parametric, nonparametric)

make_discrete(m::Semiparametric, discrete::AbstractVector{<:Bool}) =
    Semiparametric(make_discrete(m.parametric, discrete), make_discrete(m.nonparametric, discrete))

model_posterior(model::Semiparametric, data::ExperimentDataMLE; split::Bool=false) =
    model_posterior(model, data.X, data.Y, data.θ, data.length_scales, data.amplitudes, data.noise_std; split)
model_posterior(model::Semiparametric, data::ExperimentDataBI; split::Bool=false) =
    model_posterior.(Ref(model), Ref(data.X), Ref(data.Y), data.θ, data.length_scales, data.amplitudes, data.noise_std; split)

function model_posterior(
    model::Semiparametric,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    θ::AbstractVector{<:Real},
    λ::AbstractMatrix{<:Real},
    α::AbstractVector{<:Real},
    noise_std::AbstractVector{<:Real};
    split::Bool, 
)
    return model_posterior(add_mean(model.nonparametric, model.parametric(θ)), X, Y, λ, α, noise_std; split)
end

function model_loglike(model::Semiparametric, noise_std_priors::AbstractVector{<:UnivariateDistribution}, data::ExperimentData)
    function loglike(θ, λ, α, noise_std)
        ll_params = model_params_loglike(model, θ, λ, α)
        ll_data = model_data_loglike(model, θ, λ, α, noise_std, data.X, data.Y)
        ll_noise = noise_loglike(noise_std_priors, noise_std)
        return ll_params + ll_data + ll_noise
    end
end

function model_params_loglike(model::Semiparametric, θ::AbstractVector{<:Real}, λ::AbstractMatrix{<:Real}, α::AbstractVector{<:Real})
    ll_param = model_params_loglike(model.parametric, θ)
    ll_gp = model_params_loglike(model.nonparametric, λ, α)
    return ll_param + ll_gp
end

function model_data_loglike(
    model::Semiparametric,
    θ::AbstractVector{<:Real},
    λ::AbstractMatrix{<:Real},
    α::AbstractVector{<:Real},
    noise_std::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    return model_data_loglike(
        add_mean(model.nonparametric, model.parametric(θ)),
        λ,
        α,
        noise_std,
        X,
        Y,
    )
end

function sample_params(model::Semiparametric)
    θ = isempty(model.parametric.param_priors) ?
            Real[] :
            rand.(model.parametric.param_priors)
    λ = reduce(hcat, rand.(model.nonparametric.length_scale_priors))
    α = rand.(model.nonparametric.amp_priors)
    return θ, λ, α
end

param_priors(model::Semiparametric) =
    model.parametric.param_priors, model.nonparametric.length_scale_priors, model.nonparametric.amp_priors
