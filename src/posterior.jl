
"""
    average_posterior(::AbstractVector{Function}) -> Function

Return an averaged posterior predictive distribution of the given posteriors.

Useful with `ModelFitter`s which sample multiple `ModelParams` samples.
"""
average_posterior(posteriors::AbstractVector{<:Function}) =
    x -> mapreduce(p -> p(x), .+, posteriors) ./ length(posteriors)

"""
    model_posterior(::BossProblem) -> ::Function or ::AbstractVector{<:Function}

Return the posterior predictive distribution of the surrogate model with two methods;
- `post(x::AbstractVector{<:Real}) -> μs::AbstractVector{<:Real}, σs::AsbtractVector{<:Real}`
- `post(X::AbstractMatrix{<:Real}) -> μs::AbstractMatrix{<:Real}, Σs::AbstractArray{<:Real, 3}`

or a vector of such posterior functions (in case a `ModelFitter`
which samples multiple `ModelParams` has been used).

The first method takes a single point `x` of length `x_dim(::BossProblem)` from the `Domain`,
and returns the predictive means and deviations
of the corresponding output vector `y` of length `y_dim(::BossProblem)` such that:
- `μs, σs = post(x)` => `y ∼ product_distribution(Normal.(μs, σs))`
- `μs, σs = post(x)` => `y[i] ∼ Normal(μs[i], σs[i])`

The second method takes multiple points from the `Domain` as a column-wise matrix `X` of size `(x_dim, N)`,
and returns the joint predictive means and covariance matrices
of the corresponding output matrix `Y` of size `(y_dim, N)` such that:
- `μs, Σs = post(X)` => `transpose(Y) ∼ product_distribution(MvNormal.(eachcol(μs), eachslice(Σs; dims=3)))`
- `μs, Σs = post(X)` => `Y[i,:] ∼ MvNormal(μs[:,i], Σs[:,:,i])`

See also: [`model_posterior_slice`](@ref)
"""
model_posterior(prob::BossProblem) =
    model_posterior(prob.model, prob.params, prob.data)

"""
    model_posterior_slice(::BossProblem, slice::Int) -> post

Return the posterior predictive distributions of the given output `slice` with two methods:
- `post(x::AbstractVector{<:Real}) -> μ::Real, σ::Real`
- `post(X::AbstractMatrix{<:Real}) -> μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real}`

The first method takes a single point `x` of length `x_dim(::BossProblem)` from the `Domain`,
and returns the predictive mean and deviation
of the corresponding output number `y` such that:
- `μ, σ = post(x)` => `y ∼ Normal(μ, σ)`

The second method takes multiple points from the `Domain as a column-wise matrix `X` of size `(x_dim, N)`,
and returns the joint predictive mean and covariance matrix
of the corresponding output vector `y` of length `N` such that:
- `μ, Σ = post(X)` => `y ∼ MvNormal(μ, Σ)`

In case one is only interested in predicting a certain output dimension,
using `model_posterior_slice` can be more efficient than `model_posterior`
(depending on the used `SurrogateModel`).

Note that `model_posterior_slice` can be used even if `sliceable(model) == false`.
It will, however, not provide any additional efficiency in such case.

See also: [`model_posterior`](@ref)
"""
model_posterior_slice(prob::BossProblem, slice::Int) =
    model_posterior_slice(prob.model, prob.params, prob.data, slice)

# Methods to unpack `FittedParams`
model_posterior(model::SurrogateModel, params::FittedParams, data::ExperimentData) =
    model_posterior(model, get_params(params), data)
model_posterior_slice(model::SurrogateModel, params::FittedParams, data::ExperimentData, slice::Int) =
    model_posterior_slice(model, get_params(params), data, slice)

# Methods to broadcast over multiple params (due to `MultiFittedParams`)
model_posterior(model::SurrogateModel, params::AbstractVector{<:ModelParams}, data::ExperimentData) =
    model_posterior.(Ref(model), params, Ref(data))
model_posterior_slice(model::SurrogateModel, params::AbstractVector{<:ModelParams}, data::ExperimentData, slice) =
    model_posterior_slice.(Ref(model), params, Ref(data), Ref(slice))
    
# General method for surrogate models only implementing `model_posterior_slice`.
function model_posterior(model::SurrogateModel, params::ModelParams, data::ExperimentData)
    slices = model_posterior_slice.(Ref(model), Ref(params), Ref(data), 1:y_dim(data))

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
function model_posterior_slice(model::SurrogateModel, params::ModelParams, data::ExperimentData, slice::Int)
    posterior = model_posterior(model, params, data)
    
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
