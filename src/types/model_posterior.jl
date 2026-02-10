
"""
    AbstractModelPosterior{<:SurrogateModel}

An abstract model posterior. The subtypes include `ModelPosterior` and `ModelPosteriorSlice`.
"""
abstract type AbstractModelPosterior{
    M<:SurrogateModel,
} end

"""
    ModelPosterior{M<:SurrogateModel}

Contains precomputed quantities for the evaluation of the predictive posterior
of the `SurrogateModel` `M`.

Each subtype of `ModelPosterior` *should* implement:
- `mean(::ModelPosterior, ::AbstractVector{<:Real}) -> ::AbstractVector{<:Real}`
- `mean(::ModelPosterior, ::AbstractMatrix{<:Real}) -> ::AbstractMatrix{<:Real}`
- `var(::ModelPosterior, ::AbstractVector{<:Real}) -> ::AbstractVector{<:Real}`
- `var(::ModelPosterior, ::AbstractMatrix{<:Real}) -> ::AbstractMatrix{<:Real}`
- `cov(::ModelPosterior, ::AbstractMatrix{<:Real}) -> ::AbstractArray{<:Real, 3}`

and *may* implement corresponding methods:
- `mean_and_var(::ModelPosterior, ::AbstractVector{<:Real}) -> ::Tuple{...}` 
- `mean_and_var(::ModelPosterior, ::AbstractMatrix{<:Real}) -> ::Tuple{...}` 
- `mean_and_cov(::ModelPosterior, ::AbstractMatrix{<:Real}) -> ::Tuple{...}`

Additionally, the following methods are provided and *need not be implemented*:
- `std(::ModelPosterior, ::AbstractVector{<:Real}) -> ::AbstractVector{<:Real}`
- `std(::ModelPosterior, ::AbstractMatrix{<:Real}) -> ::AbstractMatrix{<:Real}`
- `mean_and_std(::ModelPosterior, ::AbstractVector{<:Real}) -> ::Tuple{...}`
- `mean_and_std(::ModelPosterior, ::AbstractMatrix{<:Real}) -> ::Tuple{...}`
- `average_mean(::AbstractVector{<:ModelPosterior}, ::AbstractVector{<:Real})`
- `average_mean(::AbstractVector{<:ModelPosterior}, ::AbstractMatrix{<:Real})`

See [`SurrogateModel`](@ref) for more information.

See also: [`ModelPosteriorSlice`](@ref)
"""
abstract type ModelPosterior{
    M<:SurrogateModel,
} <: AbstractModelPosterior{M} end

"""
    ModelPosteriorSlice{M<:SurrogateModel}

Contains precomputed quantities for the evaluation of the predictive posterior
of a single output dimension of the `SurrogateModel` `M`.

Each subtype of `ModelPosteriorSlice` *should* implement:
- `mean(::ModelPosteriorSlice, ::AbstractVector{<:Real}) -> ::Real`
- `mean(::ModelPosteriorSlice, ::AbstractMatrix{<:Real}) -> ::AbstractVector{<:Real}`
- `var(::ModelPosteriorSlice, ::AbstractVector{<:Real}) -> ::Real`
- `var(::ModelPosteriorSlice, ::AbstractMatrix{<:Real}) -> ::AbstractVector{<:Real}`
- `cov(::ModelPosteriorSlice, ::AbstractMatrix{<:Real}) -> ::AbstractMatrix{<:Real}`

and *may* implement corresponding methods:
- `mean_and_var(::ModelPosteriorSlice, ::AbstractVector{<:Real}) -> ::Tuple{...}` 
- `mean_and_var(::ModelPosteriorSlice, ::AbstractMatrix{<:Real}) -> ::Tuple{...}` 
- `mean_and_cov(::ModelPosteriorSlice, ::AbstractMatrix{<:Real}) -> ::Tuple{...}`

Additionally, the following methods are provided and *need not be implemented*:
- `std(::ModelPosteriorSlice, ::AbstractVector{<:Real}) -> ::Real`
- `std(::ModelPosteriorSlice, ::AbstractMatrix{<:Real}) -> ::AbstractVector{<:Real}`
- `mean_and_std(::ModelPosteriorSlice, ::AbstractVector{<:Real}) -> ::Tuple{...}`
- `mean_and_std(::ModelPosteriorSlice, ::AbstractMatrix{<:Real}) -> ::Tuple{...}`
- `average_mean(::AbstractVector{<:ModelPosteriorSlice}, ::AbstractVector{<:Real})`
- `average_mean(::AbstractVector{<:ModelPosteriorSlice}, ::AbstractMatrix{<:Real})`

See [`SurrogateModel`](@ref) for more information.

See also: [`ModelPosterior`](@ref)
"""
abstract type ModelPosteriorSlice{
    M<:SurrogateModel,
} <: AbstractModelPosterior{M} end


### Posterior Methods ###

# Default implementations of posterior methods are in `/src/posterior.jl`.

"""
    model_posterior(::BossProblem) -> ::Union{<:ModelPosterior, <:AbstractVector{<:ModelPosterior}}
    model_posterior(::SurrogateModel, ::ModelParams, ::ExperimentData) -> ::ModelPosterior
    model_posterior(::SurrogateModel, ::AbstractVector{<:ModelParams}, ::ExperimentData) -> ::AbstractVector{<:ModelPosterior}

Return an instance of `ModelPosterior` allowing to evaluate the posterior predictive distribution,
or a vector of `ModelPosterior`s in case of multiple sampled model parameters.

See [`ModelPosterior`](@ref) for the list of available methods.
"""
function model_posterior end

"""
    model_posterior_slice(::BossProblem, slice::Int) -> ::Union{<:ModelPosteriorSlice, <:AbstractVector{<:ModelPosteriorSlice}}
    model_posterior_slice(::SurrogateModel, ::ModelParams, ::ExperimentData, slice::Int) -> ::ModelPosteriorSlice
    model_posterior_slice(::SurrogateModel, ::AbstractVector{<:ModelParams}, ::ExperimentData, slice::Int) -> ::AbstractVector{<:ModelPosteriorSlice}

Return an instance of `ModelPosteriorSlice` allowing to evaluate the posterior predictive distribution,
or a vector of `ModelPosteriorSlice`s in case of multiple sampled model parameters.

See [`ModelPosteriorSlice`](@ref) for the list of available methods.
"""
function model_posterior_slice end

"""
    mean(::ModelPosterior, ::AbstractVector{<:Real}) -> ::AbstractVector{<:Real}
    mean(::ModelPosterior, ::AbstractMatrix{<:Real}) -> ::AbstractMatrix{<:Real}
    mean(::ModelPosteriorSlice, ::AbstractVector{<:Real}) -> ::Real
    mean(::ModelPosteriorSlice, ::AbstractMatrix{<:Real}) -> ::AbstractVector{<:Real}

Return the posterior predictive mean(s).
"""
function mean end

"""
    var(::ModelPosterior, ::AbstractVector{<:Real}) -> ::AbstractVector{<:Real}
    var(::ModelPosterior, ::AbstractMatrix{<:Real}) -> ::AbstractMatrix{<:Real}
    var(::ModelPosteriorSlice, ::AbstractVector{<:Real}) -> ::Real
    var(::ModelPosteriorSlice, ::AbstractMatrix{<:Real}) -> ::AbstractVector{<:Real}

Return the posterior predictive variance(s).
"""
function var end

"""
    cov(::ModelPosterior, ::AbstractMatrix{<:Real}) -> ::AbstractArray{<:Real, 3}
    cov(::ModelPosteriorSlice, ::AbstractMatrix{<:Real}) -> ::AbstractMatrix{<:Real}

Return the posterior predictive covariance matrix/matrices.
"""
function cov end

"""
    mean_and_var(::ModelPosterior, ::AbstractVector{<:Real}) -> ::Tuple{...}
    mean_and_var(::ModelPosterior, ::AbstractMatrix{<:Real}) -> ::Tuple{...}
    mean_and_var(::ModelPosteriorSlice, ::AbstractVector{<:Real}) -> ::Tuple{...}
    mean_and_var(::ModelPosteriorSlice, ::AbstractMatrix{<:Real}) -> ::Tuple{...}

Return the posterior predictive mean(s) and variance(s) as a tuple.

The outputs correspond exactly to the outputs of the `mean` and `var` methods,
but using `mean_and_var` can be more efficient.
"""
function mean_and_var end

"""
    mean_and_cov(::ModelPosterior, ::AbstractMatrix{<:Real}) -> ::Tuple{...}
    mean_and_cov(::ModelPosteriorSlice, ::AbstractMatrix{<:Real}) -> ::Tuple{...}

Return the posterior predictive mean(s) and covariance matrix/matrices as a tuple.

The outputs correspond exactly to the outputs of the `mean` and `cov` methods,
but using `mean_and_cov` can be more efficient.
"""
function mean_and_cov end
