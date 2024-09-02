
"""
    model_posterior_slice(::BossProblem, slice::Int) -> (x -> mean, std)

Return the posterior predictive distributions of the given `slice` output dimension.

For some models, using `model_posterior_slice` can be more efficient than `model_posterior`,
if one is only interested in the predictive distribution of a certain output dimension.

Note that `model_posterior_slice` can be used even with "nonscliceable" models.

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
    sliceable(::SurrogateModel) -> ::Bool

Returns `true` if the given surrogate model is sliceable.

See [`SurrogateModel`](@ref).
"""
sliceable(::SurrogateModel) = false

function slice(problem::BossProblem, idx::Int)
    θ_slice_ = θ_slice(problem.model, idx)
    
    return BossProblem(
        NoFitness(),
        missing,
        problem.domain,
        slice(problem.y_max, idx),
        slice(problem.model, idx),
        slice(problem.data, θ_slice_, idx),
    )
end

function slice(data::ExperimentDataPrior, θ_slice, idx::Int)
    return ExperimentDataPrior(
        data.X,
        data.Y[idx:idx,:],
    )
end
function slice(data::ExperimentDataMAP, θ_slice, idx::Int)
    return ExperimentDataMAP(
        data.X,
        data.Y[idx:idx,:],
        slice(data.params, θ_slice, idx),
        data.consistent,
    )
end
function slice(data::ExperimentDataBI, θ_slice, idx::Int)
    return ExperimentDataBI(
        data.X,
        data.Y[idx:idx,:],
        slice.(data.params, Ref(θ_slice), Ref(idx)),
        data.consistent,
    )
end

function slice(params::ModelParams, θ_slice, idx::Int)
    θ, λ, α, noise_std = params
    θ_ = slice(θ, θ_slice)
    λ_ = slice(λ, idx)
    α_ = slice(α, idx)
    noise_std_ = slice(noise_std, idx)
    params_ = θ_, λ_, α_, noise_std_
    return params_
end

slice(M::AbstractMatrix, idx::Int) = M[:,idx:idx]
slice(v::AbstractVector, idx::Int) = v[idx:idx]

slice(v::AbstractVector, slice::Nothing) = empty(v)
slice(v::AbstractVector, slice::UnitRange{<:Int}) = v[slice]
