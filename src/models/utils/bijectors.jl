
truncate(b::Bijector, mask::AbstractVector{Bool}) = b
truncate(b::Stacked, mask::AbstractVector{Bool}) = Stacked(b.bs[mask])


# Softplus

softplus(x) = log(one(x) + exp(x))
inv_softplus(x) = log(exp(x) - one(x))

sigmoid(x) = (one(x) / (one(x) + exp(-x))) # = d_softplus(x)
d_inv_softplus(x) = (one(x) / (one(x) - exp(-x)))

struct Softplus <: Bijector end

Bijectors.transform(::Softplus, x::Real) = softplus(x)
Bijectors.transform(::Softplus, x::AbstractVector{<:Real}) = softplus.(x)
Bijectors.transform!(::Softplus, x::AbstractVector{<:Real}) = (x .= softplus.(x))

Bijectors.logabsdetjac(::Softplus, x::Real) = log(abs(sigmoid(x)))
Bijectors.logabsdetjac(::Softplus, x::AbstractVector{<:Real}) = mapreduce(sigmoid, *, x) |> abs |> log

InverseFunctions.inverse(::Softplus) = InvSoftplus()

struct InvSoftplus <: Bijector end

Bijectors.transform(::InvSoftplus, x::Real) = inv_softplus(x)
Bijectors.transform(::InvSoftplus, x::AbstractVector{<:Real}) = inv_softplus.(x)
Bijectors.transform!(::InvSoftplus, x::AbstractVector{<:Real}) = (x .= inv_softplus.(x))

Bijectors.logabsdetjac(::InvSoftplus, x::Real) = log(abs(d_inv_softplus(x)))
Bijectors.logabsdetjac(::InvSoftplus, x::AbstractVector{<:Real}) = mapreduce(d_inv_softplus, *, x) |> abs |> log

InverseFunctions.inverse(::InvSoftplus) = Softplus()
