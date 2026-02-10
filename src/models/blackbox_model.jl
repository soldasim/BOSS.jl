
"""
    BlackboxModel(predict)
    BlackboxModel(; kwargs...)

Create a black-box surrogate model that uses the provided `predict` function to make predictions.
The model is pre-trained and does not require fitting to data.

The `predict` function should implement the following methods:
```julia
predict(x::AbstractVector{<:Real}, data::ExperimentData) -> (μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real})
predict(X::AbstractMatrix{<:Real}, data::ExperimentData) -> (μs::AbstractMatrix{<:Real}, σs::AbstractMatrix{<:Real})
```
where `μ` is the predicted mean and `σ` is the predicted standard deviation for each output dimension.

## Keywords
- `predict::Function`: A function that takes an input vector and experiment data,
    and returns the predicted mean and standard deviation.
- `discrete::Union{Nothing, AbstractVector{Bool}}`: A vector of booleans indicating
    which dimensions of `x` are discrete. If `discrete = nothing`, all dimensions are continuous.
    Defaults to `nothing`.
"""
@kwdef struct BlackboxModel <: SurrogateModel
    predict::Function
    discrete::Union{Nothing, AbstractVector{Bool}} = nothing
end
BlackboxModel(predict::Function) = BlackboxModel(; predict)

"""
    BlackboxParams()

The parameters of the [`BlackboxModel`](@ref).

Since the blackbox model is pre-trained, it has no parameters to fit.
This is an empty structure used to conform to the `SurrogateModel` API.
"""
struct BlackboxParams <: ModelParams{BlackboxModel} end

"""
    BlackboxPosterior

The posterior predictive distribution for the [`BlackboxModel`](@ref).

## Fields
- `predict::Function`: The prediction function from the blackbox model.
- `data::ExperimentData`: The current experiment data.
- `discrete::Union{Nothing, AbstractVector{Bool}}`: Discrete dimension indicators.
"""
struct BlackboxPosterior <: ModelPosterior{BlackboxModel}
    predict::Function
    data::ExperimentData
    discrete::Union{Nothing, AbstractVector{Bool}}
end


### SurrogateModel API Implementation ###

## Utility Methods

make_discrete(model::BlackboxModel, discrete::AbstractVector{Bool}) =
    BlackboxModel(model.predict, discrete)


## Posterior Methods

function model_posterior(model::BlackboxModel, params::BlackboxParams, data::ExperimentData)
    return BlackboxPosterior(model.predict, data, model.discrete)
end

function mean(post::BlackboxPosterior, x::AbstractVector{<:Real})
    x = discrete_round(post.discrete, x)
    μ, σ = post.predict(x, post.data)
    return μ # ::AbstractVector{<:Real}
end

function mean(post::BlackboxPosterior, X::AbstractMatrix{<:Real})
    X = hcat([discrete_round(post.discrete, x) for x in eachcol(X)]...)
    μs, σs = post.predict(X, post.data)
    return μs # ::AbstractMatrix{<:Real}
end

function var(post::BlackboxPosterior, x::AbstractVector{<:Real})
    x = discrete_round(post.discrete, x)
    μ, σ = post.predict(x, post.data)
    return σ .^ 2 # ::AbstractVector{<:Real}
end

function var(post::BlackboxPosterior, X::AbstractMatrix{<:Real})
    X = hcat([discrete_round(post.discrete, x) for x in eachcol(X)]...)
    μs, σs = post.predict(X, post.data)
    return σs .^ 2 # ::AbstractMatrix{<:Real}
end

function mean_and_var(post::BlackboxPosterior, x::AbstractVector{<:Real})
    x = discrete_round(post.discrete, x)
    μ, σ = post.predict(x, post.data)
    return μ, σ .^ 2 # ::Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}
end

function mean_and_var(post::BlackboxPosterior, X::AbstractMatrix{<:Real})
    X = hcat([discrete_round(post.discrete, x) for x in eachcol(X)]...)
    μs, σs = post.predict(X, post.data)
    return μs, σs .^ 2 # ::Tuple{<:AbstractMatrix{<:Real}, <:AbstractMatrix{<:Real}}
end


## Parameter Methods

# Blackbox models are pre-trained and have no parameters to fit.
# We return trivial implementations that satisfy the API.

function data_loglike(model::BlackboxModel, data::ExperimentData)
    # Since there are no parameters to fit, we return a constant function
    function ll_data(params::BlackboxParams)
        return 0.0
    end
end

function params_loglike(model::BlackboxModel)
    # No parameters means no prior
    function ll_params(params::BlackboxParams)
        return 0.0
    end
end

function _params_sampler(model::BlackboxModel)
    # Return a function that always returns the empty params
    function sample(rng::AbstractRNG)
        return BlackboxParams()
    end
end

function vectorizer(model::BlackboxModel)
    # No parameters to vectorize
    function vectorize(params::BlackboxParams)
        return Float64[]
    end
    
    function devectorize(params::BlackboxParams, ps::AbstractVector{<:Real})
        @assert isempty(ps)
        return BlackboxParams()
    end

    return vectorize, devectorize
end

function bijector(model::BlackboxModel)
    # No parameters means identity bijector on empty vector
    return NoBijector()
end

