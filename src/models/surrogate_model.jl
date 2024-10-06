
"""
An abstract type for a surrogate model approximating the objective function.

Example usage: `struct CustomModel <: SurrogateModel ... end`

All models *should* implement:
- `make_discrete(model::CustomModel, discrete::AbstractVector{<:Bool}) -> discrete_model::CustomModel`
- `model_posterior(model::CustomModel, data::ExperimentDataMAP) -> (x -> mean, std)`
- `model_loglike(model::CustomModel, data::ExperimentData) -> (::ModelParams -> ::Real)`
- `sample_params(model::CustomModel) -> ::ModelParams`
- `param_priors(model::CustomModel) -> ::ParamPriors`

Models *may* implement:
- `sliceable(::CustomModel) -> ::Bool`
- `model_posterior_slice(model::CustomModel, data::ExperimentDataMAP, slice::Int) -> (x -> mean, std)`

If `sliceable(::CustomModel) == true`, then the model *should* additionally implement:
- `model_loglike_slice(model::SliceableModel, data::ExperimentData, slice::Int) -> (::ModelParams -> ::Real)`
- `θ_slice(model::SliceableModel, idx::Int) -> Union{Nothing, UnitRange{<:Int}}`

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

# Broadcast over hyperparameter samples
model_posterior(model::SurrogateModel, data::ExperimentDataBI) =
    model_posterior.(Ref(model), eachsample(data))

# Broadcast over hyperparameter samples
model_posterior_slice(model::SurrogateModel, data::ExperimentDataBI, slice::Int) =
    model_posterior_slice.(Ref(model), eachsample(data), Ref(slice))

# Unspecialized method which brings no computational advantage over `model_posterior`.
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

"""
Return an averaged posterior predictive distribution of the given posteriors.

The posterior is a function `predict(x) -> (mean, std)`
which gives the mean and standard deviation of the predictive distribution as a function of `x`.
"""
average_posterior(posteriors::AbstractVector{<:Function}) =
    x -> mapreduce(p -> p(x), .+, posteriors) ./ length(posteriors)
