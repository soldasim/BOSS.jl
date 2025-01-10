
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
