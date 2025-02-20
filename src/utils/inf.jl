
"""
An auxiliary type to allow dispatch on infinity.
"""
struct Infinity <: Real end

Base.isinf(::Infinity) = true
Base.convert(::Type{F}, ::Infinity) where {F<:AbstractFloat} = F(Inf)
Base.promote_rule(::Type{F}, ::Type{Infinity}) where {F<:AbstractFloat} = F

# This method is a workaround to avoid NaNs returned from autodiff.
# See: https://github.com/soldasim/BOSS.jl/issues/2
for D in subtypes(UnivariateDistribution)
    @eval Distributions.cdf(::$D, ::Infinity) = 1.
end

# To fix ambiguities detected by Aqua.jl;
Base.promote_rule(::Type{BigFloat}, ::Type{BOSS.Infinity}) = BigFloat
Distributions.cdf(::Distributions.LocationScale{T, Distributions.Discrete, D} where {T<:Real, D<:Distributions.Distribution{Distributions.Univariate, Distributions.Discrete}}, ::BOSS.Infinity) = 1.
Distributions.cdf(::Distributions.Categorical{P} where P<:Real, ::BOSS.Infinity) = 1.
Distributions.cdf(::Distributions.LocationScale{T, Distributions.Continuous, D} where {T<:Real, D<:Distributions.Distribution{Distributions.Univariate, Distributions.Continuous}}, ::BOSS.Infinity) = 1.
