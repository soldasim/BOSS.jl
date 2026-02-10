
"""
    SurrogateModel

An abstract type for a surrogate model approximating the objective function.


## Defining Custom Surrogate Model

To define a custom surrogate model, define a new subtype `struct CustomModel <: SurrogateModel ... end`
as well as the other structures and methods described below.

The inputs in square brackets `[...]` are optional and can be used to provide additional data.
It is prefferable to define the methods without the optional inputs if possible.

See the docstrings of the individual functions for more information.


### Model Posterior Methods

Each model *should* define *at least one* of the following posterior constructors:
- `model_posterior(::SurrogateModel, ::ModelParams, ::ExperimentData) -> ::ModelPosterior`
- `model_posterior_slice(::SurrogateModel, ::ModelParams, ::ExperimentData, slice::Int) -> ::ModelPosteriorSlice`

and will usually implement the corresponding posterior type(s):
- `struct CustomPosterior <: ModelPosterior{CustomModel} ... end`
- `struct CustomPosteriorSlice <: ModelPosteriorSlice{CustomModel} ... end`

However, the model may reuse a posterior type defined for a different model.

Defining both `ModelPosterior` and `ModelPosteriorSlice` is also possible,
and can be used to provide a more efficient implementation of the posterior.

Additionally, the API described in the docstring(s) of the `ModelPosterior` or/and `ModelPosteriorSlice` type(s)
must be implemented for new posterior types.


### Model Parameters Methods

Each model *should* define a new type:
- `struct CustomParams <: ModelParams{CustomModel} ... end`

Each model *should* implement the following methods used for parameter estimation:
- `data_loglike(::SurrogateModel, ::ExperimentData) -> (::ModelParams -> ::Real)`
- `params_loglike(::SurrogateModel, [::ExperimentData]) -> (::ModelParams -> ::Real)`
- `_params_sampler(::SurrogateModel, [::ExperimentData]) -> (::AbstractRNG -> ::ModelParams)`
- `vectorizer(::SurrogateModel, [::ExperimentData]) -> (vectorize, devectorize)`
    where `vectorize(::ModelParams) -> ::AbstractVector{<:Real}` and `devectorize(::ModelParams, ::AbstractVector{<:Real}) -> ::ModelParams`
- `bijector(::SurrogateModel, [::ExperimentData]) -> ::Bijectors.Transform`

Additionally, the following methods are provided and *need not be implemented*:
- `model_loglike(::SurrogateModel, ::ExperimentData) -> (::ModelParams -> ::Real)`
- `safe_model_loglike(::SurrogateModel, ::ExperimentData) -> (::ModelParams -> ::Real)`
- `safe_data_loglike(::SurrogateModel, ::ExperimentData) -> (::ModelParams -> ::Real)`
- `params_sampler(::SurrogateModel, ::ExperimentData) -> ([::AbstractRNG] -> ::ModelParams)`


### Utility Methods

Models *may* implement:
- `make_discrete(model::SurrogateModel, discrete::AbstractVector{Bool}) -> discrete_model::SurrogateModel`
- `sliceable(::SurrogateModel) = true` (defaults to `false`)

If `sliceable(::SurrogateModel) == true`, then the model *should* additionally implement:
- `slice(model::SurrogateModel, slice::Int) -> model_slice::SurrogateModel`
- `slice(params::ModelParams, slice::Int) -> params_slice::ModelParams`
- `join_slices(slices::AbstractVector{ModelParams}) -> params::ModelParams`

Defining the `SurrogateModel` as sliceable allows for significantly more efficient parameter estimation,
but is generally not possible for all models.

`SurrogateModel`s implementing `model_posterior_slice` will usually be sliceable,
whereas models implementing `model_posterior` will not, but the API does not require this.


## See Also

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

# docstring in `src/types/problem.jl`
# function slice end

"""
    join_slices(::AbstractVector{<:ModelParams}) -> ::ModelParams

Combine the given `ModelParams` slices back into a single multi-dimensional `ModelParams` instance.
"""
function join_slices end


### Parameter Methods ###

"""
    model_loglike(::SurrogateModel, ::ExperimentData) -> (::ModelParams -> ::Real)

Return a function mapping `ModelParams` to their log-likelihood according to the current data.
"""
function model_loglike(model::SurrogateModel, data::ExperimentData)
    ll_data = data_loglike(model, data)
    ll_params = params_loglike(model, data)

    function loglike(params::ModelParams)
        return ll_data(params) + ll_params(params)
    end
end

"""
    safe_model_loglike(::SurrogateModel, ::ExperimentData; options::BossOptions) -> (::ModelParams -> ::Real)

Get a safe version of the model log-likelihood function, which returns `-Inf`
in case an error occurs while evaluating the log-likelihood of the model parameters.
"""
function safe_model_loglike end

"""
    data_loglike(::SurrogateModel, ::ExperimentData) -> (::ModelParams -> ::Real)

Construct the data log-likelihood function mapping `ModelParams`
to the log-likelihood of the current `ExperimentData`.
"""
function data_loglike end

"""
    safe_data_loglike(::SurrogateModel, ::ExperimentData; options::BossOptions) -> (::ModelParams -> ::Real)

Get a safe version of the data log-likelihood function, which returns `-Inf`
in case an error occurs while evaluating the data log-likelihood.
"""
function safe_data_loglike end

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
function params_sampler(args...)
    sampler = _params_sampler(args...)
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
