
"""
    BossProblem(; kwargs...)

Defines the whole optimization problem for the BOSS algorithm.

# Problem Definition

    There is some (noisy) blackbox function `y = f(x) = f_true(x) + ϵ` where `ϵ ~ Normal`.

    We have some surrogate model `y = model(x) ≈ f_true(x)`
    describing our knowledge (or lack of it) about the blackbox function.

    We wish to find `x ∈ domain` such that `fitness(f(x))` is maximized
    while satisfying the constraints `f(x) <= y_max`.

# Keywords
- `fitness::Fitness`: The [`Fitness`](@ref) function mapping the output to a real valued score.
- `f::Union{Function, Missing}`: The objective blackbox function.
- `domain::Domain`: The [`Domain`](@ref) of the input `x`.
- `y_max::AbstractVector{<:Real}`: The constraints on the output `y`.
- `model::SurrogateModel`: The [`SurrogateModel`](@ref).
- `params::Union{FittedParams, Nothing}`: The model parameters. Defaults to `nothing`.
- `data::ExperimentData`: The data obtained by evaluating the objective function.
- `consistent::Bool`: True iff the `model_params` have been fitted using the current `data`.
        Is set to `consistent = false` after updating the dataset,
        and to `consistent = true` after re-fitting the parameters.
        Defaults to `constistent = false`.

# See Also
[`bo!`](@ref)
"""
@kwdef mutable struct BossProblem{
    F<:Any,
    M<:SurrogateModel,
}
    fitness::Fitness = NoFitness()
    f::F
    domain::Domain
    y_max::AbstractVector{<:Real} = nothing
    model::M
    params::Union{FittedParams, Nothing} = nothing
    data::ExperimentData
    consistent::Bool = false

    function BossProblem(fitness, f::F, domain, y_max, model::M, params, data, consistent) where {F,M}
        @assert x_dim(domain) == x_dim(data)
        isnothing(y_max) && (y_max = fill(Inf, y_dim(data)))
        @assert length(y_max) == y_dim(data)
        params = _init_params(params)
        problem = new{F,M}(fitness, f, domain, y_max, model, params, data, consistent)
        return _init_problem(problem)
    end
end

_init_params(params::Nothing) = nothing
_init_params(params::ModelParams) = FixedParams(params)
_init_params(params::FittedParams) = params

function _init_problem(problem::BossProblem)
    if any(problem.domain.discrete)
        problem.domain = make_discrete(problem.domain)
        problem.model = make_discrete(problem.model, problem.domain.discrete)
    end

    # part of a workaround, see '/src/utils/inf.jl'
    if any(isinf.(problem.y_max))
        problem.y_max = [isinf(c) ? Infinity() : c for c in problem.y_max]
    end

    return problem
end

"""
    x_dim(::BossProblem) -> Int

Return the input dimension of the problem.
"""
x_dim(p::BossProblem) = x_dim(p.domain)

"""
    y_dim(::BossProblem) -> Int

Return the output dimension of the problem.
"""
y_dim(p::BossProblem) = length(p.y_max)

"""
    cons_dim(::BossProblem) -> Int
    cons_dim(::Domain) -> Int

Return the output dimension of the constraint function on the input.

See [`Domain`](@ref) for more information.
"""
cons_dim(p::BossProblem) = cons_dim(p.domain)

"""
    data_count(::BossProblem) -> Int

Return the number of datapoints in the dataset.
"""
data_count(p::BossProblem) = length(p.data)

"""
    is_consistent(::BossProblem) -> Bool

Return true iff the model parameters have been fitted using the current dataset.
"""
is_consistent(p::BossProblem) = p.consistent

"""
    get_params(::BossProblem) -> ::Union{::ModelParams, AbstractVector{<:ModelParams}}

Return the fitted `ModelParams` (or a vector of `ModelParams` samples).
"""
get_params(p::BossProblem) = get_params(p.params)

"""
    slice(::BossProblem, slice::Int) -> ::BossProblem

Return a `BossProblem` for the given `slice` output dimension.

The returned `BossProblem` has a single output dimension,
`NoFitness` and `missing` objective function.
"""
function slice(problem::BossProblem, idx::Int)    
    return BossProblem(
        NoFitness(),
        missing,
        problem.domain,
        problem.y_max[idx:idx],
        slice(problem.model, idx),
        isnothing(problem.params) ? nothing : slice(problem.params, idx),
        slice(problem.data, idx),
        problem.consistent,
    )
end

"""
    update_parameters!(problem::BossProblem{F,M}, params::ModelParams{M})

Update the model parameters.
"""
function update_parameters!(problem::BossProblem{<:Any,M1}, params::FittedParams{M2}) where {M1,M2}
    @assert M1 <: M2
    problem.params = params
    problem.consistent = true
    return params
end

"""
    augment_dataset!(::BossProblem, x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    augment_dataset!(::BossProblem, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})

Add one (as vectors) or more (as matrices) datapoints to the dataset.
"""
function augment_dataset!(problem::BossProblem, X::AbstractArray{<:Real}, Y::AbstractArray{<:Real})
    problem.data = augment_dataset(problem.data, X, Y)
    problem.consistent = false
    return problem.data
end

"""
    model_posterior(::BossProblem) -> post(s)
    model_posterior(::SurrogateModel, ::ModelParams, ::ExperimentData) -> post(s)
    model_posterior(::SurrogateModel, ::FittedParams, ::ExperimentData) -> post(s)

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

"""
    average_posterior(::AbstractVector{Function}) -> Function

Return an averaged posterior predictive distribution of the given posteriors.

Useful with `ModelFitter`s which sample multiple `ModelParams` samples.
"""
average_posterior(posteriors::AbstractVector{<:Function}) =
    x -> mapreduce(p -> p(x), .+, posteriors) ./ length(posteriors)

"""
    result(problem) -> (x, y)

Return the best found point `(x, y)`.

Returns the point `(x, y)` from the dataset of the given problem
such that `y` satisfies the constraints and `fitness(y)` is maximized.
Returns nothing if the dataset is empty or if no feasible point is present.

Does not check whether `x` belongs to the domain as no exterior points
should be present in the dataset.
"""
function result(problem::BossProblem)
    X, Y = problem.data.X, problem.data.Y
    @assert size(X)[2] == size(Y)[2]
    isempty(X) && return nothing

    feasible = is_feasible.(eachcol(Y), Ref(problem.y_max))
    fitness = problem.fitness.(eachcol(Y))
    fitness[.!feasible] .= -Inf
    best = argmax(fitness)

    feasible[best] || return nothing
    return X[:,best], Y[:,best]
end
