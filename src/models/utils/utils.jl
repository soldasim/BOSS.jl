# This file contains miscellaneous utils for defining custom `SurrogateModel`s.

# aliases for better code reabaility
const ThetaPriors = AbstractVector{<:UnivariateDistribution}
const LengthscalePriors = AbstractVector{<:MultivariateDistribution}
const AmplitudePriors = AbstractVector{<:UnivariateDistribution}
const NoiseStdPriors = AbstractVector{<:UnivariateDistribution}

"""
    NoBijector()

An auxiliary bijector, which realizes the identity transformation,
but only works on empty vectors.
"""
struct NoBijector <: Bijectors.Bijector end

Bijectors.inverse(b::NoBijector) = b

function Bijectors.transform(::NoBijector, x::AbstractVector)
    @assert isempty(x)
    return x
end

Bijectors.logabsdetjac(::NoBijector, x::AbstractVector) = zero(eltype(x))

"""
    default_bijector(priors::AbstractVector{<:Distribution}) -> ::Bijectors.Transform

Create a `bijector` for a `SurrogateModel` based on the bijectors defined
for the `Distribution`s used as the model parameter priors.

`Dirac` distributions are automatically filtered out. (Parameters with `Dirac` priors are not free
and should not be included in the vectorized parameters. See `vectorizer` for more information.)
"""
function default_bijector(priors::AbstractVector{<:Distribution})
    priors_ = filter_dirac_priors(priors)
    
    isempty(priors_) && return NoBijector()

    return Stacked(
        Bijectors.bijector.(priors_),
        ranges(length.(priors_)),
    )
end
