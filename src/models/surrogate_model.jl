
"""
An abstract type for a surrogate model approximating the objective function.

Example usage: `struct CustomModel <: SurrogateModel ... end`

# The Surrogate Model API

All models *should* implement *at least one* of:
- `model_posterior(model::CustomModel, data::ExperimentDataMAP) -> (x -> mean, std)`
- `model_posterior_slice(model::CustomModel, data::ExperimentDataMAP, slice::Int) -> (x -> mean, std)`

All models *should* implement:
- `model_loglike(model::CustomModel, data::ExperimentData) -> (::ModelParams -> ::Real)`
- `sample_params(model::CustomModel) -> ::ModelParams`
- `param_priors(model::CustomModel) -> ::ParamPriors`

Models *may* implement:
- `make_discrete(model::CustomModel, discrete::AbstractVector{<:Bool}) -> discrete_model::CustomModel`
- `sliceable(::CustomModel) = true` (false by default)

If `sliceable(::CustomModel) == true`, then the model *should* additionally implement:
- `slice(model::CustomModel, slice::Int) -> model_slice::CustomModel`
- `θ_slice(model::CustomModel, idx::Int) -> Union{Nothing, UnitRange{<:Int}}`

See also:
[`LinModel`](@ref), [`NonlinModel`](@ref),
[`GaussianProcess`](@ref),
[`Semiparametric`](@ref)
"""
abstract type SurrogateModel end

"""
    sliceable(::SurrogateModel) -> ::Bool

Returns `true` if the given surrogate model is sliceable.

See [`SurrogateModel`](@ref).
"""
sliceable(::SurrogateModel) = false

# General method for surrogate models only implementing `model_posterior_slice`.
function model_posterior(model::SurrogateModel, data::ExperimentDataMAP)
    slices = model_posterior_slice.(Ref(model), Ref(data), 1:y_dim(data))

    function post(x::AbstractVector{<:Real})
        means_and_stds = [s(x) for s in slices]
        μs = first.(means_and_stds)
        σs = second.(means_and_stds)
        return μs, σs # ::Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}
    end
    function post(X::AbstractMatrix{<:Real})
        means_and_covs = [s(X) for s in slices]
        μs = reduce(hcat, first.(means_and_covs))
        Σs = reduce((a,b) -> cat(a,b; dims=3), second.(means_and_covs))
        return μs, Σs # ::Tuple{<:AbstractMatrix{<:Real}, <:AbstractArray{<:Real, 3}}
    end
    return post
end

# General method for surrogate models only implementing `model_posterior`.
function model_posterior_slice(model::SurrogateModel, data::ExperimentDataMAP, slice::Int)
    posterior = model_posterior(model, data)
    
    function post(x::AbstractVector{<:Real})
        μs, σs = posterior(x)
        μ = μs[slice]
        σ = σs[slice]
        return μ, σ # ::Tuple{<:Real, <:Real}
    end
    function post(X::AbstractMatrix{<:Real})
        μs, Σs = posterior(X)
        μ = μs[:,slice]
        Σ = Σs[:,:,slice]
        return μ, Σ # ::Tuple{<:AbstractVector{<:Real}, <:AbstractMatrix{<:Real}}
    end
end

# Broadcast over hyperparameter samples
model_posterior(model::SurrogateModel, data::ExperimentDataBI) =
    model_posterior.(Ref(model), eachsample(data))

# Broadcast over hyperparameter samples
model_posterior_slice(model::SurrogateModel, data::ExperimentDataBI, slice::Int) =
    model_posterior_slice.(Ref(model), eachsample(data), Ref(slice))

"""
Return an averaged posterior predictive distribution of the given posteriors.

The posterior is a function `predict(x) -> (mean, std)`
which gives the mean and standard deviation of the predictive distribution as a function of `x`.
"""
average_posterior(posteriors::AbstractVector{<:Function}) =
    x -> mapreduce(p -> p(x), .+, posteriors) ./ length(posteriors)
