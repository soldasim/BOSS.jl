
# Useful with broadcasting.
cond_func(f::Function) = (b, x) -> b ? f(x) : x

"""
Return true iff x belongs to the optimization domain.
"""
function in_domain(domain::AbstractBounds, x::AbstractVector)
    lb, ub = domain
    any(x .< lb) && return false
    any(x .> ub) && return false
    return true
end

"""
Return true iff `y` satisfies the given constraints.
"""
is_feasible(y::AbstractVector{<:Real}, cons::AbstractVector{<:Real}) = all(y .< cons)

x_dim(model::Nonparametric) = length(first(model.length_scale_priors))
x_dim(model::Semiparametric) = length(first(model.nonparametric.length_scale_priors))
x_dim(problem::OptimizationProblem) = length(problem.discrete)

y_dim(model::Nonparametric) = length(model.length_scale_priors)
y_dim(model::Semiparametric) = length(model.nonparametric.length_scale_priors)
y_dim(problem::OptimizationProblem) = length(problem.cons)

θ_len(model::Parametric) = length(model.param_priors)
θ_len(model::Nonparametric) = 0
θ_len(model::Semiparametric) = length(model.parametric.param_priors)

λ_len(model::Parametric) = 0
λ_len(model::Nonparametric) = y_dim(model) * x_dim(model)
λ_len(model::Semiparametric) = y_dim(model) * x_dim(model)

param_count(model::SurrogateModel) = θ_len(model) + λ_len(model)
