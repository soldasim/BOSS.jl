
"""
An abstract type for parametric surrogate models.

See also: [`LinModel`](@ref), [`NonlinModel`](@ref)
"""
abstract type Parametric <: SurrogateModel end

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
- `param_priors::AbstractVector{<:UnivariateDistribution}`: The prior distributions for
        the parameters `[θ₁₁, ..., θ₁ₚ, ..., θₘ₁, ..., θₘₚ]` according to the definition above.
- `discrete::Union{Nothing, <:AbstractVector{<:Bool}}`: Describes which dimensions are discrete.
        Typically, the `discrete` kwarg can be ignored by the end-user as it will be updated
        according to the `Domain` of the `BossProblem` during BOSS initialization.
"""
struct LinModel{
    P<:AbstractVector{<:UnivariateDistribution},
    D<:Union{Nothing, <:AbstractVector{<:Bool}},
} <: Parametric
    lift::Function
    param_priors::P
    discrete::D
end
LinModel(;
    lift,
    param_priors,
    discrete=nothing,
) = LinModel(lift, param_priors, discrete)

"""
    NonlinModel(; kwargs...)

A parametric surrogate model.

If your model is linear, you can use `LinModel` which will provide better performance in the future. (Not yet implemented.)

Define the model as `y = predict(x, θ)` where `θ` are the model parameters.

# Keywords
- `predict::Function`: The `predict` function according to the definition above.
- `param_priors::AbstractVector{<:UnivariateDistribution}`: The prior distributions for the model parameters.
- `discrete::Union{Nothing, <:AbstractVector{<:Bool}}`: Describes which dimensions are discrete.
        Typically, the `discrete` kwarg can be ignored by the end-user as it will be updated
        according to the `Domain` of the `BossProblem` during BOSS initialization.
"""
struct NonlinModel{
    P<:AbstractVector{<:UnivariateDistribution},
    D<:Union{Nothing, <:AbstractVector{<:Bool}},
} <: Parametric
    predict::Function
    param_priors::P
    discrete::D
end
NonlinModel(;
    predict,
    param_priors,
    discrete=nothing,
) = NonlinModel(predict, param_priors, discrete)

(model::Parametric)(θ::AbstractVector{<:Real}) = x -> model(x, θ)

function (model::LinModel)(x::AbstractVector{<:Real}, θ::AbstractVector{<:Real})
    x = discrete_round(model.discrete, x)

    ϕs = model.lift(x)
    m = length(ϕs)

    ϕ_lens = length.(ϕs)
    θ_indices = vcat(0, partial_sums(ϕ_lens))
    
    y = [(θ[θ_indices[i]+1:θ_indices[i+1]])' * ϕs[i] for i in 1:m]
    return y
end

function partial_sums(array::AbstractArray)
    isempty(array) && return empty(array)
    s = zero(first(array))
    return [(s += val) for val in array]
end

function (m::NonlinModel)(x::AbstractVector{<:Real}, θ::AbstractVector{<:Real})
    x = discrete_round(m.discrete, x)
    return m.predict(x, θ)
end

Base.convert(::Type{NonlinModel}, model::LinModel) =
    NonlinModel(
        (x, θ) -> model(x, θ),
        model.param_priors,
        model.discrete,
    )

make_discrete(m::LinModel, discrete::AbstractVector{<:Bool}) =
    LinModel(m.lift, m.param_priors, discrete)
make_discrete(m::NonlinModel, discrete::AbstractVector{<:Bool}) =
    NonlinModel(m.predict, m.param_priors, discrete)

model_posterior(model::Parametric, data::ExperimentDataMAP; split::Bool=false) =
    model_posterior(model, data.θ, data.noise_std; split)
model_posterior(model::Parametric, data::ExperimentDataBI; split::Bool=false) = 
    model_posterior.(Ref(model), data.θ, data.noise_std; split)

function model_posterior(
    model::Parametric,
    θ::AbstractVector{<:Real},
    noise_std::AbstractVector{<:Real};
    split::Bool,
)
    if split
        return [(x) -> (model(x, θ)[i], noise_std[i]) for i in eachindex(noise_std)]
    else
        return (x) -> (model(x, θ), noise_std)
    end
end

function model_loglike(model::Parametric, noise_std_priors::AbstractVector{<:UnivariateDistribution}, data::ExperimentData)
    function loglike(θ, λ, α, noise_std)
        ll_params = model_params_loglike(model, θ)
        ll_data = model_data_loglike(model, θ, noise_std, data.X, data.Y)
        ll_noise = noise_loglike(noise_std_priors, noise_std)
        return ll_params + ll_data + ll_noise
    end
end

function model_params_loglike(model::Parametric, θ::AbstractVector{<:Real})
    return mapreduce(p -> logpdf(p...), +, zip(model.param_priors, θ); init=0.)
end

function model_data_loglike(
    model::Parametric,
    θ::AbstractVector{<:Real},
    noise_std::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    ll_datum(x, y) = logpdf(MvNormal(model(x, θ), noise_std), y)
    return mapreduce(d -> ll_datum(d...), +, zip(eachcol(X), eachcol(Y)))
end

function sample_params(model::Parametric)
    θ = isempty(model.param_priors) ?
            Real[] :
            rand.(model.param_priors)
    λ = Real[;;]
    α = Real[]
    return θ, λ, α
end

param_priors(model::Parametric) =
    model.param_priors, MultivariateDistribution[], UnivariateDistribution[]
