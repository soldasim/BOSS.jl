
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
Return an averaged posterior predictive distribution of the given posteriors.

The posterior is a function `predict(x) -> (mean, std)`
which gives the mean and standard deviation of the predictive distribution as a function of `x`.
"""
average_posteriors(posteriors::AbstractVector{<:Function}) =
    x -> mapreduce(p -> p(x), .+, posteriors) ./ length(posteriors)

noise_loglike(noise_std_priors::AbstractVector{<:UnivariateDistribution}, noise_std::AbstractVector{<:Real}) =
    sum(logpdf.(noise_std_priors, noise_std))

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
