
"""
Return an Inverse Gamma distribution
with _approximately_ 0.99 probability mass between `lb` and `ub.`
"""
function calc_inverse_gamma(lb, ub)
    @assert (lb >= 0) && (ub >= 0)
    @assert lb < ub
    μ = (ub + lb) / 2
    σ = (ub - lb) / 6
    a = (μ^2 / σ^2) + 2
    b = μ * ((μ^2 / σ^2) + 1)
    return InverseGamma(a, b)
end

# This constructor is deprecated in Distributions.jl
mvnormal(μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}) =
    MvNormal(μ, Diagonal(map(abs2, σ)))

# This constructor is deprecated in Distributions.jl
mvlognormal(μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}) =
    MvLogNormal(μ, Diagonal(map(abs2, σ)))


# --- Truncated multivariate normal distribution ---

"""
    TruncatedMvNormal(μ, Σ, lb, ub)

Defines the truncated multivariate normal distribution with mean `μ`, covariance matrix `Σ`,
lower bounds `lb`, and upper bounds `ub`.
"""
@kwdef struct TruncatedMvNormal{
    D, #::Bool
} <: ContinuousMultivariateDistribution
    μ::AbstractVector{<:Real}
    Σ::AbstractMatrix{<:Real}
    lb::AbstractVector{<:Real}
    ub::AbstractVector{<:Real}

    function TruncatedMvNormal(μ, Σ, lb, ub)
        @warn "The `logpdf` of `TruncatedMvNormal` is only valid up to a constant." maxlog=1
        diag = isdiag(Σ)
        return new{diag}(μ, Σ, lb, ub)
    end
end

Base.length(d::TruncatedMvNormal) = length(d.μ)
# Distributions.sampler(d::TruncatedMvNormal)
# Base.eltype(d::TruncatedMvNormal)

Base.minimum(d::TruncatedMvNormal) = d.lb
Base.maximum(d::TruncatedMvNormal) = d.ub

# simple rejection sampling
function Distributions._rand!(rng::AbstractRNG, d::TruncatedMvNormal{false}, x::AbstractVector{<:Real})    
    Distributions._rand!(rng, MvNormal(d.μ, d.Σ), x)
    while _out_of_bounds(d, x)
        Distributions._rand!(rng, MvNormal(d.μ, d.Σ), x)
    end
    return x
end
function Distributions._rand!(rng::AbstractRNG, d::TruncatedMvNormal{true}, x::AbstractVector{<:Real})
    for i in eachindex(x)
        x[i] = rand(rng, truncated(Normal(d.μ[i], d.Σ[i,i]); lower=d.lb[i], upper=d.ub[i]))
    end
    return x
end

function _out_of_bounds(d::TruncatedMvNormal, x::AbstractVector{<:Real})
    for i in eachindex(x)
        if x[i] < d.lb[i] || x[i] > d.ub[i]
            return true
        end
    end
    return false
end

# return logpdf of the non-truncated MvNormal - valid up to an additive constant
function Distributions._logpdf(d::TruncatedMvNormal, x::AbstractVector{<:Real})
    if _out_of_bounds(d, x)
        return -Inf
    end
    return Distributions._logpdf(MvNormal(d.μ, d.Σ), x)
end

function Bijectors.bijector(d::TruncatedMvNormal)
    return Bijectors.TruncatedBijector(d.lb, d.ub)
end
