
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
