
model_posterior(prob::BossProblem) =
    model_posterior(prob.model, get_params(prob), prob.data)

model_posterior_slice(prob::BossProblem, slice::Int) =
    model_posterior_slice(prob.model, get_params(prob), prob.data, slice)

model_posterior(model::SurrogateModel, params::FittedParams, data::ExperimentData) =
    model_posterior(model, get_params(params), data)

model_posterior_slice(model::SurrogateModel, params::FittedParams, data::ExperimentData, slice::Int) =
    model_posterior_slice(model, get_params(params), data, slice)

# Broadcast over multiple model parameters (due `MultiFittedParams`).
model_posterior(model::SurrogateModel, params::AbstractVector{<:ModelParams}, data::ExperimentData) =
    model_posterior.(Ref(model), params, Ref(data))

model_posterior_slice(model::SurrogateModel, params::AbstractVector{<:ModelParams}, data::ExperimentData, slice::Int) =
    model_posterior_slice.(Ref(model), params, Ref(data), Ref(slice))


### Default `ModelPosterior` ###

"""
    DefaultModelPosterior{<:SurrogateModel, <:ModelPosteriorSlice}
    post = model_posterior(::SurrogateModel, ::ModelParams, ::ExperimentData)

The default implementatoin of the `ModelPosterior` used for `SurrogateModel`s
that only implement a custom `ModelPosteriorSlice`.
"""
struct DefaultModelPosterior{
    M<:SurrogateModel,
    P<:ModelPosteriorSlice{M},
} <: ModelPosterior{M}
    slices::Vector{P}
end

function model_posterior(model::SurrogateModel, params::ModelParams, data::ExperimentData)
    slices = [model_posterior_slice(model, params, data, i) for i in 1:y_dim(data)]
    return DefaultModelPosterior(slices)
end

function mean(post::DefaultModelPosterior, x::AbstractVector{<:Real})
    return mean.(post.slices, Ref(x)) # ::AbstractVector{<:Real}
end
function mean(post::DefaultModelPosterior, X::AbstractMatrix{<:Real})
    return vcat([mean(s, X)' for s in post.slices]...) # ::AbstractMatrix{<:Real}
end

function var(post::DefaultModelPosterior, x::AbstractVector{<:Real})
    return var.(post.slices, Ref(x)) # ::AbstractVector{<:Real}
end
function var(post::DefaultModelPosterior, X::AbstractMatrix{<:Real})
    return vcat([var(s, X)' for s in post.slices]...) # ::AbstractMatrix{<:Real}
end

function cov(post::DefaultModelPosterior, X::AbstractMatrix{<:Real})
    return cat(cov.(post.slices, Ref(X))...; dims=3) # ::AbstractArray{<:Real, 3}
end

function mean_and_var(post::DefaultModelPosterior, x::AbstractVector{<:Real})
    μs_and_σs = mean_and_var.(post.slices, Ref(x))
    μs = first.(μs_and_σs)
    σs = second.(μs_and_σs)
    return μs, σs # ::Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}
end
function mean_and_var(post::DefaultModelPosterior, X::AbstractMatrix{<:Real})
    μs_and_σs = mean_and_var.(post.slices, Ref(X))
    μs = vcat([first(x)' for x in μs_and_σs]...)
    σs = vcat([second(x)' for x in μs_and_σs]...)
    return μs, σs # ::Tuple{<:AbstractMatrix{<:Real}, <:AbstractMatrix{<:Real}}
end

function mean_and_cov(post::DefaultModelPosterior, X::AbstractMatrix{<:Real})
    μs_and_Σs = mean_and_cov.(post.slices, Ref(X))
    μs = vcat([first(x)' for x in μs_and_Σs]...)
    Σs = cat(second.(μs_and_Σs)...; dims=3)
    return μs, Σs # ::Tuple{<:AbstractMatrix{<:Real}, <:AbstractArray{<:Real, 3}}
end

# No `predictive_samples(post::DefaultModelPosterior, ...)` redirection to `post.slices` is defined
# here on purpose: bundling independently-modeled per-dimension slices into a joint `(D, K)` tuple
# would silently assert a dependency structure across dimensions that the model never modeled.
# Contrast with `DefaultModelPosteriorSlice`'s `predictive_samples` below, which redirects the
# other way (joint -> one dimension) and is safe. See `predictive_samples`'s docstring (in
# `types/model_posterior.jl`) for the full reasoning.

### Default `ModelPosteriorSlice` ###

"""
    DefaultModelPosteriorSlice{<:SurrogateModel, <:ModelPosterior}
    post_slice = model_posterior_slice(::SurrogateModel, ::ModelParams, ::ExperimentData, slice::Int)

The default implementatoin of the `ModelPosteriorSlice` used for `SurrogateModel`s
that only implement a custom `ModelPosterior`.
"""
struct DefaultModelPosteriorSlice{
    M<:SurrogateModel,
    P<:ModelPosterior{M},
} <: ModelPosteriorSlice{M}
    post::P
    idx::Int
end

function model_posterior_slice(model::SurrogateModel, params::ModelParams, data::ExperimentData, slice::Int)
    post = model_posterior(model, params, data)
    return DefaultModelPosteriorSlice(post, slice)
end

function mean(post::DefaultModelPosteriorSlice, x::AbstractVector{<:Real})
    μ = mean(post.post, x)
    return μ[post.idx] # ::Real
end
function mean(post::DefaultModelPosteriorSlice, X::AbstractMatrix{<:Real})
    μs = mean(post.post, X)
    return μs[post.idx, :] # ::AbstractVector{<:Real}
end

function var(post::DefaultModelPosteriorSlice, x::AbstractVector{<:Real})
    σ = var(post.post, x)
    return σ[post.idx] # ::Real
end
function var(post::DefaultModelPosteriorSlice, X::AbstractMatrix{<:Real})
    σs = var(post.post, X)
    return σs[post.idx, :] # ::AbstractVector{<:Real}
end

function cov(post::DefaultModelPosteriorSlice, X::AbstractMatrix{<:Real})
    Σs = cov(post.post, X)
    return Σs[:,:,post.idx] # ::AbstractMatrix{<:Real}
end

function mean_and_var(post::DefaultModelPosteriorSlice, x::AbstractVector{<:Real})
    μ, σ = mean_and_var(post.post, x)
    return μ[post.idx], σ[post.idx] # ::Tuple{<:Real, <:Real}
end
function mean_and_var(post::DefaultModelPosteriorSlice, X::AbstractMatrix{<:Real})
    μs, σs = mean_and_var(post.post, X)
    return μs[post.idx, :], σs[post.idx, :] # ::Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}
end

function mean_and_cov(post::DefaultModelPosteriorSlice, X::AbstractMatrix{<:Real})
    μs, Σs = mean_and_cov(post.post, X)
    return μs[post.idx, :], Σs[:,:,post.idx] # ::Tuple{<:AbstractVector{<:Real}, <:AbstractMatrix{<:Real}}
end

# Redirects to the joint `post.post`'s `predictive_samples` and reads off row `post.idx` — the
# safe direction (marginalizing a genuine joint sample down to one dimension is always valid),
# unlike `DefaultModelPosterior`'s (deliberately absent) reverse redirection above.
function predictive_samples(post::DefaultModelPosteriorSlice, x::AbstractVector{<:Real})
    Ys, Ws = predictive_samples(post.post, x) # Ys: (D, K), Ws: (1, K) -- see `predictive_samples`'s docstring
    return Ys[post.idx, :], vec(Ws) # ::Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}, both length K
end
function predictive_samples(post::DefaultModelPosteriorSlice, X::AbstractMatrix{<:Real})
    Ys, Ws = predictive_samples(post.post, X) # Ys: (D, K, n), Ws: (1, K, n)
    return Ys[post.idx, :, :], Ws[1, :, :] # ::Tuple{<:AbstractMatrix{<:Real}, <:AbstractMatrix{<:Real}}, both (K, n)
end


### Other default posterior methods ###

function mean_and_var(post::ModelPosterior, X::AbstractVecOrMat{<:Real})
    μ = mean(post, X)
    σ2 = var(post, X)
    return μ, σ2
end

function mean_and_cov(post::ModelPosterior, X::AbstractMatrix{<:Real})
    μs = mean(post, X)
    Σs = cov(post, X)
    return μs, Σs
end

function std(post::AbstractModelPosterior, X::AbstractVecOrMat{<:Real})
    return var(post, X) .|> sqrt
end

function mean_and_std(post::AbstractModelPosterior, X::AbstractVecOrMat{<:Real})
    μs, σs = mean_and_var(post, X)
    return μs, sqrt.(σs)
end

function average_mean(posts::AbstractVector{<:AbstractModelPosterior}, X::AbstractVecOrMat{<:Real})
    return mean.(posts, Ref(X)) |> mean
end
