
"""
    ModelParamsPrior(::SurrogateModel, ::ExperimentData)
    ModelParamsPrior(::SurrogateModel, ::ModelParams, ::ExperimentData)

A subtype of `ContinuousMultivariateDistribution` that represents the joint prior distribution
of all model parameters.

The parameter prior is already completely defined by the methods `_params_sampler` and `params_loglike`
of the given `SurrogateModel`. This is just a convenience structure implementing the `Distribution` API.
"""
struct ModelParamsPrior <: ContinuousMultivariateDistribution
    sample::Function
    loglike::Function
    bijector::Any
    length::Int
    eltype::Type
end
function ModelParamsPrior(model::SurrogateModel, data::ExperimentData)
    sampler = params_sampler(model, data)
    vectorize, devectorize = vectorizer(model, data)

    params = sampler()
    ps = vectorize(params)

    sample(rng) = sampler(rng) |> vectorize
    
    ll_params_ = params_loglike(model, data)
    loglike(ps) = ll_params_(devectorize(params, ps))

    b = bijector(model, data)
    
    return ModelParamsPrior(sample, loglike, b, length(ps), eltype(ps))
end

Base.length(d::ModelParamsPrior) = d.length
Base.eltype(d::ModelParamsPrior) = d.eltype

function Distributions._rand!(rng::AbstractRNG, d::ModelParamsPrior, x::AbstractVector{<:Real})
    x .= d.sample(rng)
end
function Distributions._logpdf(d::ModelParamsPrior, x::AbstractVector{<:Real})
    return d.loglike(x)
end

Bijectors.bijector(d::ModelParamsPrior) = d.bijector
