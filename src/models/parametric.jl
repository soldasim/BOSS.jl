
"""
An abstract type for parametric surrogate models.

See also: [`LinModel`](@ref), [`NonlinModel`](@ref)
"""
abstract type Parametric{
    N<:Union{Nothing, <:NoiseStdPriors},
} <: SurrogateModel end

(model::Parametric)(θ::AbstractVector{<:Real}) = x -> model(x, θ)

"""
    LinModel(; kwargs...)

A parametric surrogate model linear in its parameters.

This model definition will provide better performance than the more general 'NonlinModel' in the future.
This feature is not implemented yet so it is equivalent to using `NonlinModel` for now.

The linear model is defined as
```
    ϕs = lift(x)
    y = [θs[i]' * ϕs[i] for i in 1:m]
```
where
```
    x = [x₁, ..., xₙ]
    y = [y₁, ..., yₘ]
    θs = [θ₁, ..., θₘ], θᵢ = [θᵢ₁, ..., θᵢₚ]
    ϕs = [ϕ₁, ..., ϕₘ], ϕᵢ = [ϕᵢ₁, ..., ϕᵢₚ]
```
and ``n, m, p ∈ R``.

# Keywords
- `lift::Function`: Defines the `lift` function `(::Vector{<:Real}) -> (::Vector{Vector{<:Real}})`
        according to the definition above.
- `theta_priors::AbstractVector{<:UnivariateDistribution}`: The prior distributions for
        the parameters `[θ₁₁, ..., θ₁ₚ, ..., θₘ₁, ..., θₘₚ]` according to the definition above.
- `noise_std_priors::NoiseStdPriors`: The prior distributions
        of the noise standard deviations of each `y` dimension.
"""
@kwdef struct LinModel{
    P<:AbstractVector{<:UnivariateDistribution},
    D<:Union{Nothing, <:AbstractVector{<:Bool}},
    N<:Union{Nothing, <:NoiseStdPriors},
} <: Parametric{N}
    lift::Function
    theta_priors::P
    discrete::D = nothing
    noise_std_priors::N = nothing
end

"""
    NonlinModel(; kwargs...)

A parametric surrogate model.

If your model is linear, you can use `LinModel` which will provide better performance in the future. (Not yet implemented.)

Define the model as `y = predict(x, θ)` where `θ` are the model parameters.

# Keywords
- `predict::Function`: The `predict` function according to the definition above.
- `theta_priors::AbstractVector{<:UnivariateDistribution}`: The prior distributions for the model parameters.
- `noise_std_priors::NoiseStdPriors`: The prior distributions
        of the noise standard deviations of each `y` dimension.
"""
@kwdef struct NonlinModel{
    P<:AbstractVector{<:UnivariateDistribution},
    D<:Union{Nothing, <:AbstractVector{<:Bool}},
    N<:Union{Nothing, <:NoiseStdPriors},
} <: Parametric{N}
    predict::Function
    theta_priors::P
    discrete::D = nothing
    noise_std_priors::N = nothing
end

function (model::LinModel)(x::AbstractVector{<:Real}, θ::AbstractVector{<:Real})
    x = discrete_round(model.discrete, x)

    ϕs = model.lift(x)
    m = length(ϕs)

    ϕ_lens = length.(ϕs)
    θ_indices = vcat(0, cumsum(ϕ_lens))
    
    y = [(θ[θ_indices[i]+1:θ_indices[i+1]])' * ϕs[i] for i in 1:m]
    return y
end

function (m::NonlinModel)(x::AbstractVector{<:Real}, θ::AbstractVector{<:Real})
    x = discrete_round(m.discrete, x)
    return m.predict(x, θ)
end

Base.convert(::Type{NonlinModel}, model::LinModel) =
    NonlinModel(
        (x, θ) -> model(x, θ),
        model.theta_priors,
        model.discrete,
        model.noise_std_priors,
    )

remove_noise_priors(m::LinModel) =
    LinModel(m.lift, m.theta_priors, m.discrete, nothing)
remove_noise_priors(m::NonlinModel) =
    NonlinModel(m.predict, m.theta_priors, m.discrete, nothing)

make_discrete(m::LinModel, discrete::AbstractVector{<:Bool}) =
    LinModel(m.lift, m.theta_priors, discrete, m.noise_std_priors)
make_discrete(m::NonlinModel, discrete::AbstractVector{<:Bool}) =
    NonlinModel(m.predict, m.theta_priors, discrete, m.noise_std_priors)

function model_posterior(model::Parametric, data::ExperimentDataMAP)
    θ, _, _, noise_std = data.params
    return model_posterior(model, θ, noise_std)
end

function model_posterior(
    model::Parametric,
    θ::AbstractVector{<:Real},
    noise_std::AbstractVector{<:Real},
)
    function post(x::AbstractVector{<:Real})
        μs = model(x, θ)
        σs = noise_std
        return μs, σs # ::Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}
    end
    function post(X::AbstractMatrix{<:Real})
        count = size(X)[2]
        μs = mapreduce(x -> model(x, θ)', vcat, eachcol(X))
        Σs = mapreduce(σ -> Diagonal(fill(σ^2, count)), (a,b) -> cat(a,b; dims=3), noise_std)
        return μs, Σs
    end
    return post
end

function model_loglike(model::Parametric, data::ExperimentData)
    function loglike(params)
        ll_data = data_loglike(model, data.X, data.Y, params)
        ll_params = model_params_loglike(model, params)
        return ll_data + ll_params
    end
end

function data_loglike(
    model::Parametric,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    params::ModelParams,
)
    θ, λ, α, noise_std = params
    ll_datum(x, y) = logpdf(mvnormal(model(x, θ), noise_std), y)
    return mapreduce(d -> ll_datum(d...), +, zip(eachcol(X), eachcol(Y)))
end

function model_params_loglike(model::Parametric, params::ModelParams)
    θ, λ, α, noise_std = params
    ll_theta = sum(logpdf.(model.theta_priors, θ); init=0.)
    ll_noise = sum(logpdf.(model.noise_std_priors, noise_std))
    return ll_theta + ll_noise
end

function sample_params(model::Parametric)
    θ = isempty(model.theta_priors) ?
            Float64[] :
            rand.(model.theta_priors)
    λ = nothing
    α = nothing
    noise_std = rand.(model.noise_std_priors)
    return θ, λ, α, noise_std
end

function param_priors(model::Parametric)
    θ_priors = model.theta_priors
    λ_priors = nothing
    α_priors = nothing
    noise_std_priors = model.noise_std_priors
    @assert !isnothing(noise_std_priors)
    return θ_priors, λ_priors, α_priors, noise_std_priors
end
