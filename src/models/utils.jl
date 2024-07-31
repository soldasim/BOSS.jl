
"""
    model_posterior(::BossProblem) -> (x -> mean, std)

Return the posterior predictive distribution of the Gaussian Process.

The posterior is a function `predict(x) -> (mean, std)`
which gives the mean and std of the predictive distribution as a function of `x`.

See also: [`model_posterior_slice`](@ref)
"""
model_posterior(prob::BossProblem) =
    model_posterior(prob.model, prob.data)

# Broadcast over hyperparameter samples
model_posterior(model::SurrogateModel, data::ExperimentDataBI) =
    model_posterior.(Ref(model), eachsample(data))

"""
    model_posterior_slice(::BossProblem, slice::Int) -> (x -> mean, std)

Return the posterior predictive distributions of the given `slice` output dimension.

In case of a Gaussian process model (or a semiparametric model),
using `model_posterior_slice` is more efficient than `model_posterior`
if one is only interested in the predictive distribution of a certain output dimension.

See also: [`model_posterior`](@ref)
"""
model_posterior_slice(prob::BossProblem, slice::Int) =
    model_posterior_slice(prob.model, prob.data, slice)

# Broadcast over hyperparameter samples
model_posterior_slice(model::SurrogateModel, data::ExperimentDataBI, slice::Int) =
    model_posterior_slice.(Ref(model), eachsample(data), Ref(slice))

# Unspecialized method which brings no computational advantage over `model_posterior`.
function model_posterior_slice(model::SurrogateModel, data::ExperimentDataMAP, slice::Int)
    posterior = model_posterior(model, data)
    
    function post(x::AbstractVector{<:Real})
        μ, std = posterior(x)
        return μ[slice], std[slice]
    end
    function post(X::AbstractMatrix{<:Real})
        μ, std = posterior(X)
        return μ[slice,:], std[slice,:]
    end
end

"""
Return an averaged posterior predictive distribution of the given posteriors.

The posterior is a function `predict(x) -> (mean, std)`
which gives the mean and standard deviation of the predictive distribution as a function of `x`.
"""
average_posteriors(posteriors::AbstractVector{<:Function}) =
    x -> mapreduce(p -> p(x), .+, posteriors) ./ length(posteriors)

discrete_round(::Nothing, x::AbstractVector{<:Real}) = x
discrete_round(::Missing, x::AbstractVector{<:Real}) = round.(x)
discrete_round(dims::AbstractVector{<:Bool}, x::AbstractVector{<:Real}) = cond_func(round).(dims, x)

noise_loglike(noise_std_priors, noise_std) = mapreduce(p -> logpdf(p...), +, zip(noise_std_priors, noise_std))

function sample_params(model::SurrogateModel, noise_std_priors::AbstractVector{<:UnivariateDistribution})
    model_params = sample_params(model)
    noise_std = rand.(noise_std_priors)
    return model_params..., noise_std
end

function param_shapes(model::SurrogateModel)
    θ_priors, λ_priors, α_priors = param_priors(model)
    θ_shape = (length(θ_priors),)
    λ_shape = isempty(λ_priors) ?
        (0, 0) :
        (length(first(λ_priors)), length(λ_priors))
    α_shape = (length(α_priors),)
    return θ_shape, λ_shape, α_shape
end

param_counts(model::SurrogateModel) = prod.(param_shapes(model))
