using InteractiveUtils
using Distributions

# Useful with broadcasting.
cond_func(f::Function) = (b, x) -> b ? f(x) : x

"""
    in_domain(x, domain) -> Bool

Return true iff x belongs to the domain.
"""
function in_domain(x::AbstractVector{<:Real}, domain::Domain)
    in_bounds(x, domain.bounds) || return false
    isnothing(domain.cons) && return true
    return all(domain.cons(x) .>= 0.)
end

"""
    in_bounds(x, bounds) -> Bool

Return true iff x belongs to the given bounds.
"""
function in_bounds(x::AbstractVector{<:Real}, bounds::AbstractBounds)
    lb, ub = bounds
    any(x .< lb) && return false
    any(x .> ub) && return false
    return true
end

"""
    is_feasible(y, y_max) -> Bool

Return true iff `y` satisfies the given constraints.
"""
is_feasible(y::AbstractVector{<:Real}, y_max::AbstractVector{<:Real}) = all(y .<= y_max)

x_dim(model::Nonparametric) = length(first(model.length_scale_priors))
x_dim(model::Semiparametric) = length(first(model.nonparametric.length_scale_priors))
x_dim(domain::Domain) = length(domain.discrete)
x_dim(problem::OptimizationProblem) = length(problem.domain.discrete)

y_dim(model::Nonparametric) = length(model.length_scale_priors)
y_dim(model::Semiparametric) = length(model.nonparametric.length_scale_priors)
y_dim(problem::OptimizationProblem) = length(problem.y_max)

θ_len(model::Parametric) = length(model.param_priors)
θ_len(model::Nonparametric) = 0
θ_len(model::Semiparametric) = length(model.parametric.param_priors)

λ_len(model::Parametric) = 0
λ_len(model::Nonparametric) = y_dim(model) * x_dim(model)
λ_len(model::Semiparametric) = y_dim(model) * x_dim(model)

param_count(model::SurrogateModel) = θ_len(model) + λ_len(model)

cons_dim(domain::Domain) = isnothing(domain.cons) ? 0 : length(domain.cons(mean(domain.bounds)))
cons_dim(problem::OptimizationProblem) = cons_dim(problem.domain)

"""
    result(problem) -> (x, y)

Return the best found point `(x, y)`.

Returns the point `(x, y)` from the dataset of the given problem
such that `y` satisfies the constraints and `fitness(y)` is maximized.
Returns nothing if the dataset is empty or if no feasible point is present.

Does not check whether `x` belongs to the domain as no exterior points
should be present in the dataset.
"""
function result(problem::OptimizationProblem)
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
