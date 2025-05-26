
"""
    SurrogateModel

An abstract type for a surrogate model approximating the objective function.

# Defining Custom Surrogate Model

To define a custom surrogate model,
define a new subtype of `SurrogateModel` and a new subtype of `ModelParams`:
- `struct CustomModel <: SurrogateModel ... end`
- `struct ModelParams <: ModelParams{CustomModel} ... end`

The following methods should be implemented for the new `CustomModel` and `CustomModelParams` types.
The input parameters in square brackets (e.g. `[::ExperimentData]`) are optional.
It is preferrable to omit them if it is possible for your model.
See the docs of the individual functions for more information.

All models *should* implement *at least one* of:
- `model_posterior(model::SurrogateModel, params::ModelParams, data::ExperimentData) -> (x -> mean, std)`
- `model_posterior_slice(model::SurrogateModel, params::ModelParams, data::ExperimentData, slice::Int) -> (x -> mean, std)`

All models *should* implement:
- `data_loglike(::SurrogateModel, ::ExperimentData) -> (::ModelParams -> ::Real)`
- `params_loglike(::SurrogateModel, [::ExperimentData]) -> (::ModelParams -> ::Real)`
- `_params_sampler(::SurrogateModel, [::ExperimentData]) -> (::AbstractRNG -> ::ModelParams)`
- `vectorizer(::SurrogateModel, [::ExperimentData]) -> (vectorize, devectorize)`
    where `vectorize(::ModelParams) -> ::AbstractVector{<:Real}` and `devectorize(::ModelParams, ::AbstractVector{<:Real}) -> ::ModelParams`
- `bijector(::SurrogateModel, [::ExperimentData]) -> ::Bijectors.Transform`

Models *may* implement:
- `make_discrete(model::SurrogateModel, discrete::AbstractVector{Bool}) -> discrete_model::SurrogateModel`
- `sliceable(::SurrogateModel) = true` (false by default)

If `sliceable(::SurrogateModel) == true`, then the model *should* additionally implement:
- `slice(model::SurrogateModel, slice::Int) -> model_slice::SurrogateModel`
- `slice(params::ModelParams, slice::Int) -> params_slice::ModelParams`
- `join_slices(slices::AbstractVector{ModelParams}) -> params::ModelParams`

# See Also

[`LinearModel`](@ref), [`NonlinearModel`](@ref),
[`GaussianProcess`](@ref),
[`Semiparametric`](@ref)
"""
abstract type SurrogateModel end

"""
    ModelParams{M<:SurrogateModel}

Contains all parameters of the `SurrogateModel` `M`.

See [`SurrogateModel`](@ref) for more information.
"""
abstract type ModelParams{
    M<:SurrogateModel,
} end

Base.length(p::ModelParams) = length(vectorize(p))

"""
    model_posterior(::SurrogateModel, ::ModelParams, ::ExperimentData) -> post(s)
    model_posterior(::SurrogateModel, ::FittedParams, ::ExperimentData) -> post(s)


"""
function model_posterior end

function model_posterior_slice end

"""
    model_loglike(::SurrogateModel, ::ExperimentData) -> (::ModelParams -> ::Real)

Return a function mapping `ModelParams` to their log-likelihood according to the current data.
"""
function model_loglike end

function model_loglike(model::SurrogateModel, data::ExperimentData)
    ll_data = data_loglike(model, data)
    ll_params = params_loglike(model, data)

    function loglike(params::ModelParams)
        return ll_data(params) + ll_params(params)
    end
end

"""
    data_loglike(::SurrogateModel, ::ExperimentData) -> (::ModelParams -> ::Real)

Construct the data log-likelihood function mapping `ModelParams`
to the log-likelihood of the current `ExperimentData`.
"""
function data_loglike end

"""
    params_loglike(::SurrogateModel, [::ExperimentData]) -> (::ModelParams -> ::Real)

Construct the model parameters log-likelihood function mapping `ModelParams`
to their log-likelihood.

The parameters returned by the (@ref)[`params_sampler`] should be sampled
exactly according to this log-likelihood.
"""
function params_loglike end

params_loglike(model::SurrogateModel, data::ExperimentData) = params_loglike(model)

"""
    params_sampler(::SurrogateModel, [::ExperimentData]) -> ([::AbstractRNG] -> ::ModelParams)

Return a function (or a callable structure) which samples `ModelParams` from their *prior* distributions.
(I.e. the sampling is *not* conditioned on the data.)

The parameters are sampled exactly according to the log-likelihood
defined by the `params_loglike` function.

This is a user-facing function. Implement `_params_sampler` instead
when defining a custom `SurrogateModel`.
"""
function params_sampler(model::SurrogateModel, data::ExperimentData)
    sampler = _params_sampler(model, data)
    
    sample() = sampler(Random.default_rng())
    sample(rng::AbstractRNG) = sampler(rng)

    return sample
end

"""
    _params_sampler(::SurrogateModel, [::ExperimentData]) -> (::AbstractRNG -> ::ModelParams)

Return a function (or a callable structure) which samples `ModelParams` from their *prior* distributions.
(I.e. the sampling is *not* conditioned on the data.)

The parameters should be sampled exactly according to the log-likelihood
defined by the `params_loglike` function.

This is an internal function used as a part of the `SurrogateModel` API.
Use `params_sampler` to sample model parameters instead.
"""
function _params_sampler end

_params_sampler(model::SurrogateModel, data::ExperimentData) = _params_sampler(model)

"""
    vectorizer(::SurrogateModel, [::ExperimentData]) -> (vectorize, devectorize)

    vectorize(::ModelParams) -> ::AbstractVector{<:Real}
    devectorize(::ModelParams, ::AbstractVector{<:Real}) -> ::ModelParams

Return two functions (or callable structures) which transform `ModelParams` to a vector and back.

The first function `vectorize` transforms an instance of `ModelParams`
to a real-valued vector containing only the *free* parameters of the model.

The second function `devectorize` transforms the vectorized parameters
back to an instance of `ModelParams`. The `ModelParams` instance provided
as the first argument is not changed. It is only used to determine the original shapes
of the parameters, copy fixed parameters, and/or copy other metadata.

`params == devectorize(params_, vectorize(params))` should hold even if `params_ != params`.

Be sure to check parameter priors for `Dirac` priors
and exclude such parameters from the vector as they are not free.
"""
function vectorizer end

vectorizer(model::SurrogateModel, data::ExperimentData) = vectorizer(model)

"""
    bijector(::SurrogateModel, [::ExperimentData]) -> ::Bijectors.Transform

Return a `Bijectors.Transform` which maps the vectorized model parameters to ``\\mathbb{R}^n``
and its inverse maps them back to their (possibly constrained) domain.

Note that the bijector is applied to the *vectorized* parameters (only containing free parameters).

Consider using the (@ref)[`default_bijector`] function to infer the bijector from the parameter priors.
"""
function bijector end

bijector(model::SurrogateModel, data::ExperimentData) = bijector(model)

"""
    make_discrete(::SurrogateModel, discrete::AbstractVector{Bool}) -> ::SurrogateModel

Return a new instance of the given `SurrogateModel` with discretized input dimensions
according to the given `discrete` vector.

Defining a `make_discrete` method for a `SurrogateModel` subtype enables its use
with discrete or mixed `Domain`s.
"""
function make_discrete end

"""
    sliceable(::SurrogateModel) -> ::Bool

Returns `true` if the given surrogate model is sliceable along the output dimension.

Making a `SurrogateModel` subtype sliceable allows for a more efficient MAP estimation of its parameters.
"""
sliceable(::SurrogateModel) = false

"""
    slice(::SurrogateModel, ::Int) -> ::SurrogateModel
    slice(::ModelParams, ::Int) -> ::ModelParams

Return a new instance of the given `SurrogateModel` or `ModelParams` containing
a single-dimensional of the given object corresponding to the specified output dimension.
"""
function slice end

"""
    join_slices(::AbstractVector{<:ModelParams}) -> ::ModelParams

Combine the given `ModelParams` slices back into a single multi-dimensional `ModelParams` instance.
"""
function join_slices end
