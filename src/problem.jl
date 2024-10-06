
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
- `fitness::Fitness`: The [`Fitness`](@ref) function.
- `f::Union{Function, Missing}`: The objective blackbox function.
- `domain::Domain`: The [`Domain`](@ref) of `x`.
- `y_max`: The constraints on `y`. (See the definition above.)
- `model::SurrogateModel`: The [`SurrogateModel`](@ref).
- `data::ExperimentData`: The initial data of objective function evaluations.
        See [`ExperimentDataPrior`].

See also: [`bo!`](@ref)
"""
mutable struct BossProblem{
    F<:Any,
}
    fitness::Fitness
    f::F
    domain::Domain
    y_max::AbstractVector{<:Real}
    model::SurrogateModel
    data::ExperimentData
end
BossProblem(;
    fitness = NoFitness(),
    f,
    domain,
    model,
    data,
    y_max = fill(Inf, y_dim(data)),
) = BossProblem(fitness, f, domain, y_max, model, data)

"""
    slice(::BossProblem, slice::Int) -> ::BossProblem

Return a `BossProblem` for the given `slice` output dimension.

The returned `BossProblem` has a single output dimension,
`NoFitness` and `missing` objective function.
"""
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

"""
    model_posterior(::BossProblem) -> (x -> mean, std)

Return the posterior predictive distribution of the surrogate model.

The posterior is a function with two methods:
- `predict(x::AbstractVector{<:Real}) -> (mean::AbstractVector{<:Real}, std)`
which gives the mean and std of the predictive distribution as a function of `x`.

See also: [`model_posterior_slice`](@ref)
"""
model_posterior(prob::BossProblem) =
    model_posterior(prob.model, prob.data)

"""
    model_posterior_slice(::BossProblem, slice::Int) -> (x -> mean, std)

Return the posterior predictive distributions of the given `slice` output dimension.

For some models, using `model_posterior_slice` can be more efficient than `model_posterior`,
if one is only interested in the predictive distribution of a certain output dimension.

Note that `model_posterior_slice` can be used even if `sliceable(model) == false`,
it will just be less efficient.

See also: [`model_posterior`](@ref)
"""
model_posterior_slice(prob::BossProblem, slice::Int) =
    model_posterior_slice(prob.model, prob.data, slice)

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
