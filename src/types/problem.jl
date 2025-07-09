
"""
    BossProblem(; kwargs...)

Defines the whole optimization problem for the BOSS algorithm.

## Problem Definition

There is some (noisy) blackbox function `y = f(x) = f_true(x) + ϵ` where `ϵ ~ Normal`.

We wish to find `x ∈ domain` such that `fitness(f(x))` is maximized
while satisfying the constraints `f(x) <= y_max`.

## Keywords

The following keywords correspond to all fields of the `BossProblem` type.

The keywords marked by "(*)" are required. Note that at least a single initial data point
must be provided to initialize the `BossProblem`.

- (*) `f::Union{Function, Missing}`: The objective blackbox function.
- (*) `domain::Domain`: The [`Domain`](@ref) of the input `x`.
- `y_max::AbstractVector{<:Real}`: The constraints on the output `y`.
- (*) `acquisition::AcquisitionFunction`: The acquisition function used to select
        the next evaluation point in each iteration. Usually contains the `fitness` function.
- (*) `model::SurrogateModel`: The [`SurrogateModel`](@ref).
- `params::Union{FittedParams, Nothing}`: The fitted model parameters. Defaults to `nothing`.
- (*) `data::ExperimentData`: The data obtained by evaluating the objective function.
- `consistent::Bool`: True iff the `model_params` have been fitted using the current `data`.
        Is set to `consistent = false` after updating the dataset,
        and to `consistent = true` after re-fitting the parameters.
        Defaults to `constistent = false`.

## See Also

[`bo!`](@ref)
"""
@kwdef mutable struct BossProblem{
    F<:Any,
}
    f::F
    domain::Domain
    y_max::AbstractVector{<:Real} = nothing
    acquisition::Union{AcquisitionFunction, Missing}
    model::SurrogateModel
    params::Union{FittedParams, Nothing} = nothing
    data::ExperimentData
    consistent::Bool = false

    function BossProblem(f::F, domain, y_max, acquisition, model, params, data, consistent) where {F}
        @assert x_dim(domain) == x_dim(data)
        isnothing(y_max) && (y_max = fill(Inf, y_dim(data)))
        @assert length(y_max) == y_dim(data)
        params = _init_params(params)
        problem = new{F}(f, domain, y_max, acquisition, model, params, data, consistent)
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
    get_fitness(::BossProblem) -> (y -> ::Real)

Return the fitness function is it is defined for the given `BossProblem`.
Otherwise, throw `MethodError`.
"""
get_fitness(p::BossProblem) = get_fitness(p.acquisition)

"""
    get_params(::BossProblem) -> ::Union{::ModelParams, AbstractVector{<:ModelParams}}
    get_params(::UniFittedParams) -> ::ModelParams
    get_params(::MultiFittedParams) -> ::AbstractVector{<:ModelParams}

Return the fitted `ModelParams` or a vector of `ModelParams` samples.
"""
get_params(p::BossProblem) = get_params(p.params)

"""
    slice(::BossProblem, slice::Int) -> ::BossProblem

Return a `BossProblem` for the given `slice` output dimension.

The returned `BossProblem` has a single output dimension,
`missing` objective function, and `missing` acquisition function.
"""
function slice(problem::BossProblem, idx::Int)    
    return BossProblem(
        missing,
        problem.domain,
        problem.y_max[idx:idx],
        missing,
        slice(problem.model, idx),
        isnothing(problem.params) ? nothing : slice(problem.params, idx),
        slice(problem.data, idx),
        problem.consistent,
    )
end

"""
    update_parameters!(problem::BossProblem, params::ModelParams)

Update the model parameters.
"""
function update_parameters!(problem::BossProblem, params::FittedParams{M2}) where {M2}
    @assert typeof(problem.model) <: M2
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
