
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

