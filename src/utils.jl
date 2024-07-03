
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
Return the posterior predictive distribution of the Gaussian Process.

The posterior is a function `predict(x) -> (mean, std)`
which gives the mean and std of the predictive distribution as a function of `x`.
"""
model_posterior(prob::BossProblem) =
    model_posterior(prob.model, prob.data)

"""
Exclude points exterior to the given `x` domain from the given `X` and `Y` data matrices
and return new matrices `X_` and `Y_`.
"""
function exclude_exterior_points(domain::Domain, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}; options::BossOptions)
    @assert size(X)[2] == size(Y)[2]
    datasize = size(X)[2]

    exterior = fill(false, datasize)
    for i in 1:datasize
        in_domain(X[:,i], domain) || (exterior[i] = true)
    end

    options.info && any(exterior) && @warn "Some data are exterior to the domain and will be discarded!"
    all(exterior) && return eltype(X)[;;], eltype(Y)[;;]
    
    X_ = reduce(hcat, (X[:,i] for i in 1:datasize if !exterior[i]))[:,:]
    Y_ = reduce(hcat, (Y[:,i] for i in 1:datasize if !exterior[i]))[:,:]
    return X_, Y_
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
function update_parameters!(::Type{MLE}, data::ExperimentDataPrior,
    θ,
    length_scales,
    amplitudes,
    noise_std,
)
    return ExperimentDataMLE(
        data.X,
        data.Y,
        θ,
        length_scales,
        amplitudes,
        noise_std,
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
    return data
end
function update_parameters!(::Type{T}, data::ExperimentDataPost,
    θ,
    length_scales,
    amplitudes,
    noise_std,
) where {T<:ModelFit}
    throw(ArgumentError("Inconsistent experiment data!\nCannot switch from MLE to BI or vice-versa."))
end

"""
Add one (as vectors) or more (as matrices) datapoints to the dataset.
"""
function add_data!(data::ExperimentData, X::AbstractArray{<:Real}, Y::AbstractArray{<:Real})
    data.X = hcat(data.X, X)
    data.Y = hcat(data.Y, Y)
    return data
end
