using InteractiveUtils
using Distributions

# Useful with broadcasting.
cond_func(f::Function) = (b, x) -> b ? f(x) : x

"""
Return true iff x belongs to the domain.
"""
function in_domain(domain::Domain, x::AbstractVector{<:Real})
    in_bounds(domain.bounds, x) || return false
    isnothing(domain.cons) && return true
    return all(domain.cons(x) .>= 0.)
end

"""
Return true iff x belongs to the given bounds.
"""
function in_bounds(bounds::AbstractBounds, x::AbstractVector{<:Real})
    lb, ub = bounds
    any(x .< lb) && return false
    any(x .> ub) && return false
    return true
end

"""
Return true iff `y` satisfies the given constraints.
"""
is_feasible(y::AbstractVector{<:Real}, cons::AbstractVector{<:Real}) = all(y .<= cons)

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

function exclude_exterior_points(domain::Domain, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}; info::Bool)
    @assert size(X)[2] == size(Y)[2]
    datasize = size(X)[2]

    exterior = fill(false, datasize)
    for i in 1:datasize
        in_domain(domain, X[:,i]) || (exterior[i] = true)
    end

    info && any(exterior) && @warn "Some data are exterior to the domain and will be discarded!"
    all(exterior) && return eltype(X)[;;], eltype(Y)[;;]
    
    X_ = reduce(hcat, (X[:,i] for i in 1:datasize if !exterior[i]))
    Y_ = reduce(hcat, (Y[:,i] for i in 1:datasize if !exterior[i]))
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
