
# Useful with broadcasting.
cond_func(f::Function) = (b, x) -> b ? f(x) : x

"""
    in_domain(x, domain) -> Bool

Return true iff x belongs to the domain.
"""
function in_domain(x::AbstractVector{<:Real}, domain::Domain)
    in_bounds(x, domain.bounds) || return false
    in_discrete(x, domain.discrete) || return false
    in_cons(x, domain.cons) || return false
    return true
end

function in_bounds(x::AbstractVector{<:Real}, bounds::AbstractBounds)
    lb, ub = bounds
    any(x .< lb) && return false
    any(x .> ub) && return false
    return true
end

in_discrete(x::AbstractVector{<:Real}, discrete::AbstractVector{<:Bool}) =
    all(round.(x[discrete]) .== x[discrete])

in_cons(x::AbstractVector{<:Real}, cons::Nothing) = true
in_cons(x::AbstractVector{<:Real}, cons) = all(cons(x) .>= 0.)

"""
    is_feasible(y, y_max) -> Bool

Return true iff `y` satisfies the given constraints.
"""
is_feasible(y::AbstractVector{<:Real}, y_max::AbstractVector{<:Real}) = all(y .<= y_max)

x_dim(domain::Domain) = length(domain.discrete)
x_dim(problem::BossProblem) = length(problem.domain.discrete)

y_dim(problem::BossProblem) = length(problem.y_max)

cons_dim(domain::Domain) = isnothing(domain.cons) ? 0 : length(domain.cons(mean(domain.bounds)))
cons_dim(problem::BossProblem) = cons_dim(problem.domain)

params_total(problem::BossProblem) = sum(param_counts(problem.model)) + y_dim(problem)

function data_count(problem::BossProblem)
    xsize = size(problem.data.X)[2]
    ysize = size(problem.data.Y)[2]
    @assert xsize == ysize
    return xsize
end

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

"""
Exclude points exterior to the given `x` domain from the given `X` and `Y` data matrices
and return new matrices `X_` and `Y_`.
"""
function exclude_exterior_points(domain::Domain, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}; options::BossOptions=BossOptions())
    interior = in_domain.(eachcol(X), Ref(domain))
    all(interior) && return X, Y
    options.info && @warn "Some data are exterior to the domain and will be discarded!"
    return X[:,interior], Y[:,interior]
end

"""
An auxiliary type to allow dispatch on infinity.
"""
struct Infinity <: Real end

Base.isinf(::Infinity) = true
Base.convert(::Type{F}, ::Infinity) where {F<:AbstractFloat} = F(Inf)
Base.promote_rule(::Type{F}, ::Type{Infinity}) where {F<:AbstractFloat} = F

# This method is a workaround to avoid NaNs returned from autodiff.
# See: https://github.com/Sheld5/BOSS.jl/issues/2
for D in subtypes(UnivariateDistribution)
    @eval Distributions.cdf(::$D, ::Infinity) = 1.
end

"""
Update model parameters.
"""
function update_parameters!(::Type{MAP}, data::ExperimentDataPrior,
    θ,
    length_scales,
    amplitudes,
    noise_std,
)
    return ExperimentDataMAP(
        data.X,
        data.Y,
        θ,
        length_scales,
        amplitudes,
        noise_std,
        true,  # consistent
    )
end
function update_parameters!(::Type{BI}, data::ExperimentDataPrior,
    θ,
    length_scales,
    amplitudes,
    noise_std,
)
    return ExperimentDataBI(
        data.X,
        data.Y,
        θ,
        length_scales,
        amplitudes,
        noise_std,
        true,  # consistent
    )
end
function update_parameters!(::Type{T}, data::ExperimentDataPost{T},
    θ,
    length_scales,
    amplitudes,
    noise_std,
) where {T<:ModelFit}
    data.θ = θ
    data.length_scales = length_scales
    data.amplitudes = amplitudes
    data.noise_std = noise_std
    data.consistent = true
    return data
end
function update_parameters!(::Type{T}, data::ExperimentDataPost,
    θ,
    length_scales,
    amplitudes,
    noise_std,
) where {T<:ModelFit}
    throw(ArgumentError("Inconsistent experiment data!\nCannot switch from MAP to BI or vice-versa."))
end

"""
    augment_dataset!(data::ExperimentDataPost, x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    augment_dataset!(data::ExperimentDataPost, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})

Add one (as vectors) or more (as matrices) datapoints to the dataset.
"""
function augment_dataset!(data::ExperimentDataPost, X::AbstractArray{<:Real}, Y::AbstractArray{<:Real})
    data.X = hcat(data.X, X)
    data.Y = hcat(data.Y, Y)
    data.consistent = false
    return data
end

"""
    consistent(::ExperimentData)

Returns true if the parameters in the data are estimated
using to the current dataset (X, Y).

Returns false if the dataset have been augmented
since the parameters have been estimated.
"""
consistent(data::ExperimentDataPrior) = false
consistent(data::ExperimentDataPost) = data.consistent

"""
    eachsample(::ExperimentData)

Return a `BISamples` iterator over the hyperparameter samples contained in the data.
"""
eachsample(data::ExperimentDataBI) = BISamples(data)

"""
    get_sample(::ExperimentData, ::Int)

Return a single hyperparameter sample as an instance of `ExperimentDataMAP`.
"""
function get_sample(data::ExperimentDataBI, idx::Int)
    return ExperimentDataMAP(
        data.X,
        data.Y,
        isnothing(data.θ) ? nothing : data.θ[idx],
        isnothing(data.length_scales) ? nothing : data.length_scales[idx],
        isnothing(data.amplitudes) ? nothing : data.amplitudes[idx],
        data.noise_std[idx],
        data.consistent,
    )
end

"""
    sample_count(::ExperimentDataBI)

Return the number of hyperparameter samples stored in the data.
"""
function sample_count(data::ExperimentDataBI)
    len = length(data.noise_std)
    check_len(p) = length(p) == len
    check_len(::Nothing) = true
    @assert check_len.((data.θ, data.length_scales, data.amplitudes)) |> all
    return len
end

"""
Iterator over BI samples contained in `ExperimentDataBI` structure.
The returned elements are instances of `ExperimentDataMAP`.
"""
struct BISamples
    data::ExperimentDataBI
    length::Int
end
BISamples(data::ExperimentDataBI) = BISamples(data, sample_count(data))

Base.getindex(samples::BISamples, idx::Int) = get_sample(samples.data, idx)
Base.length(samples::BISamples) = samples.length
Base.eltype(::BISamples) = ExperimentDataMAP

function Base.iterate(samples::BISamples)
    if samples.length == 0
        return nothing
    else
        return samples[1], 1
    end
end
function Base.iterate(samples::BISamples, state::Int)
    state += 1
    if state > samples.length
        return nothing
    else
        return samples[state], state
    end
end
