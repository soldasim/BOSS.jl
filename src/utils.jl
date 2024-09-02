
"""
    ith(i)(collection) == collection[i]
    ith(i).(collections) == [c[i] for c in collections]
"""
ith(i::Int) = (x) -> x[i]

"""
    cond_func(f)(x, b) == (b ? f(x) : x)
    conf_func(f).(xs, bs) == [b ? f(x) : x for (b,x) in zip(bs,xs)]
"""
cond_func(f::Function) = (x, b) -> b ? f(x) : x

discrete_round(::Nothing, x::AbstractVector{<:Real}) = x
discrete_round(::Missing, x::AbstractVector{<:Real}) = round.(x)
discrete_round(dims::AbstractVector{<:Bool}, x::AbstractVector{<:Real}) = cond_func(round).(x, dims)

"""
    is_feasible(y, y_max) -> Bool

Return true iff `y` satisfies the given constraints.
"""
is_feasible(y::AbstractVector{<:Real}, y_max::AbstractVector{<:Real}) = all(y .<= y_max)

x_dim(problem::BossProblem) = length(problem.domain.discrete)
x_dim(domain::Domain) = length(domain.discrete)
x_dim(data::ExperimentData) = size(data.X)[1]

y_dim(problem::BossProblem) = length(problem.y_max)
y_dim(data::ExperimentData) = size(data.Y)[1]

cons_dim(domain::Domain) = isnothing(domain.cons) ? 0 : length(domain.cons(mean(domain.bounds)))
cons_dim(problem::BossProblem) = cons_dim(problem.domain)

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

function param_shapes(priors::ParamPriors)
    θ_priors, λ_priors, α_priors, noise_std_priors = priors

    θ_shape = (length(θ_priors),)
    λ_shape = isnothing(λ_priors) ?
        nothing :
        (length(first(λ_priors)), length(λ_priors))
    α_shape = isnothing(α_priors) ?
        nothing :
        (length(α_priors),)
    noise_std_shape = (length(noise_std_priors),)

    shapes = θ_shape, λ_shape, α_shape, noise_std_shape
    return shapes
end
param_shapes(model::SurrogateModel) = param_shapes(param_priors(model))

function param_counts(priors::ParamPriors)
    shapes = param_shapes(priors)
    count(s::Nothing) = 0
    count(s::Tuple) = prod(s)
    counts = count.(shapes)
    return counts
end
param_counts(model::SurrogateModel) = param_counts(param_priors(model))

function params_total(priors::ParamPriors)
    counts = param_counts(priors)
    return sum(counts)
end
params_total(model::SurrogateModel) = params_total(param_priors(model))
