
"""
    truncate(b::Bijector, mask::AbstractVector{Bool}) -> ::Bijector
    truncate(b::Stacked, mask::AbstractVector{Bool}) -> ::Stacked

Return a new bijector that skips all parameters where `mask` is `false`.
(Only affects `Stacked` bijectors.)
"""
function truncate(b, mask::AbstractVector{Bool})
    return b
end
function truncate(b::Stacked, mask::AbstractVector{Bool})
    @assert all(b.ranges_in .== b.ranges_out)

    last_i = 0
    bs = eltype(b.bs)[]
    ranges = eltype(b.ranges_in)[]

    for idx in eachindex(b.bs)
        b_ = b.bs[idx]
        r_ = b.ranges_in[idx]
        mask_ = @view mask[r_]

        len_ = sum(mask_)
        (len_ == 0) && continue
        r_ = (last_i + 1):(last_i + len_)
        last_i += len_

        b_ = truncate(b_, mask_)

        push!(bs, b_)
        push!(ranges, r_)
    end

    return Stacked(bs, ranges)
end

"""
    simplify(b::Bijector) -> ::Bijector
    simplify(b::Stacked) -> ::Stacked

Simplify the given bijector by joining identical bijectors in `Stacked` bijectors.
"""
function simplify(b)
    return b
end
function simplify(b::Stacked)
    b = linearize(b)
    # @assert all(b.ranges_in .== b.ranges_out) # already checked in `linearize`

    last_b = nothing
    bs = eltype(b.bs)[]
    ranges = eltype(b.ranges_in)[]

    for idx in eachindex(b.bs)
        b_ = b.bs[idx]
        r_ = b.ranges_in[idx]

        if b_ == last_b
            ranges[end] = ranges[end].start:r_.stop
        else
            push!(bs, b_)
            push!(ranges, r_)
            last_b = b_
        end
    end

    return Stacked(bs, ranges)
end

"""
    linearize(b::Bijector) -> ::Bijector
    linearize(b::Stacked) -> ::Stacked

Linearize the given bijector by flattening nested `Stacked` bijectors.
"""
function linearize(b)
    return b
end
function linearize(b::Stacked)
    @assert all(b.ranges_in .== b.ranges_out)

    bs = []
    lens = []

    for idx in eachindex(b.bs)
        b_ = b.bs[idx]
        r_ = b.ranges_in[idx]

        b_ = linearize(b_)
        if b_ isa Stacked
            append!(bs, b_.bs)
            append!(lens, length.(b_.ranges_in))
        else
            push!(bs, b_)
            push!(lens, length(r_))
        end
    end

    # fix the eltypes
    bs = [bs...]
    lens = [lens...]

    return Stacked(bs, ranges(lens))
end


# --- Softplus ---

softplus(x) = log(one(x) + exp(x))
inv_softplus(x) = log(exp(x) - one(x))

d_softplus(x) = sigmoid(x)
d_inv_softplus(x) = (one(x) / (one(x) - exp(-x)))

# Softplus is a bijection from (-∞, ∞) to (0, ∞)
struct Softplus <: Bijector end

Bijectors.transform(::Softplus, x::Real) = softplus(x)
Bijectors.transform(::Softplus, x::AbstractVector{<:Real}) = softplus.(x)
Bijectors.transform!(::Softplus, x::AbstractVector{<:Real}) = (x .= softplus.(x))

Bijectors.logabsdetjac(::Softplus, x::Real) = log(abs(d_softplus(x)))
Bijectors.logabsdetjac(::Softplus, x::AbstractVector{<:Real}) = mapreduce(d_softplus, *, x) |> abs |> log

InverseFunctions.inverse(::Softplus) = InvSoftplus()

# Inverse Softplus is a bijection from (0, ∞) to (-∞, ∞)
struct InvSoftplus <: Bijector end

Bijectors.transform(::InvSoftplus, x::Real) = inv_softplus(x)
Bijectors.transform(::InvSoftplus, x::AbstractVector{<:Real}) = inv_softplus.(x)
Bijectors.transform!(::InvSoftplus, x::AbstractVector{<:Real}) = (x .= inv_softplus.(x))

Bijectors.logabsdetjac(::InvSoftplus, x::Real) = log(abs(d_inv_softplus(x)))
Bijectors.logabsdetjac(::InvSoftplus, x::AbstractVector{<:Real}) = mapreduce(d_inv_softplus, *, x) |> abs |> log

InverseFunctions.inverse(::InvSoftplus) = Softplus()


# --- Scaled Softplus ---

# Scaled Softplus is a bijection from (-∞, ∞) to (lb, ∞)
struct ScaledSoftplus{
    T<:Real,
} <: Bijector
    lb::T
end

Bijectors.transform(b::ScaledSoftplus, x::Real) = softplus(x) + b.lb
Bijectors.transform(b::ScaledSoftplus, x::AbstractVector{<:Real}) = softplus.(x) .+ b.lb
Bijectors.transform!(b::ScaledSoftplus, x::AbstractVector{<:Real}) = (x .= softplus.(x) .+ b.lb)

Bijectors.logabsdetjac(b::ScaledSoftplus, x::Real) = log(abs(d_softplus(x)))
Bijectors.logabsdetjac(b::ScaledSoftplus, x::AbstractVector{<:Real}) = mapreduce(d_softplus, *, x) |> abs |> log

InverseFunctions.inverse(b::ScaledSoftplus) = ScaledInvSoftplus(b.lb)


# Scaled Inverse Softplus is a bijection from (lb, ∞) to (-∞, ∞)
struct ScaledInvSoftplus{
    T<:Real,
} <: Bijector
    lb::T
end

Bijectors.transform(b::ScaledInvSoftplus, x::Real) = inv_softplus(x - b.lb)
Bijectors.transform(b::ScaledInvSoftplus, x::AbstractVector{<:Real}) = inv_softplus.(x .- b.lb)
Bijectors.transform!(b::ScaledInvSoftplus, x::AbstractVector{<:Real}) = (x .= inv_softplus.(x .- b.lb))

Bijectors.logabsdetjac(b::ScaledInvSoftplus, x::Real) = log(abs(d_inv_softplus(x)))
Bijectors.logabsdetjac(b::ScaledInvSoftplus, x::AbstractVector{<:Real}) = mapreduce(d_inv_softplus, *, x) |> abs |> log

InverseFunctions.inverse(b::ScaledInvSoftplus) = ScaledSoftplus(b.lb)


# --- Sigmoid & Logit ---

sigmoid(x) = (one(x) / (one(x) + exp(-x)))
logit(x) = log(x / (one(x) - x))

d_sigmoid(x) = sigmoid(x) * (one(x) - sigmoid(x))
d_logit(x) = (one(x) / (x * (one(x) - x)))

# Sigmoid is a bijection from (-∞, ∞) to (0, 1)
struct Sigmoid <: Bijector end

Bijectors.transform(::Sigmoid, x::Real) = sigmoid(x)
Bijectors.transform(::Sigmoid, x::AbstractVector{<:Real}) = sigmoid.(x)
Bijectors.transform!(::Sigmoid, x::AbstractVector{<:Real}) = (x .= sigmoid.(x))

Bijectors.logabsdetjac(::Sigmoid, x::Real) = log(abs(d_sigmoid(x)))
Bijectors.logabsdetjac(::Sigmoid, x::AbstractVector{<:Real}) = mapreduce(d_sigmoid, *, x) |> abs |> log

InverseFunctions.inverse(::Sigmoid) = Logit()

# Logit is a bijection from (0, 1) to (-∞, ∞)
struct Logit <: Bijector end

Bijectors.transform(::Logit, x::Real) = logit(x)
Bijectors.transform(::Logit, x::AbstractVector{<:Real}) = logit.(x)
Bijectors.transform!(::Logit, x::AbstractVector{<:Real}) = (x .= logit.(x))

Bijectors.logabsdetjac(::Logit, x::Real) = log(abs(d_logit(x)))
Bijectors.logabsdetjac(::Logit, x::AbstractVector{<:Real}) = mapreduce(d_logit, *, x) |> abs |> log

InverseFunctions.inverse(::Logit) = Sigmoid()


# --- Scaled Sigmoid & Logit ---

scaled_sigmoid(x, lb, ub) = (ub - lb) * sigmoid(x) + lb
scaled_logit(x, lb, ub) = logit((x - lb) / (ub - lb))

d_scaled_sigmoid(x, lb, ub) = (ub - lb) * d_sigmoid(x)
d_scaled_logit(x, lb, ub) = (one(x) / (ub - lb)) * d_logit((x - lb) / (ub - lb))

# Scaled Sigmoid is a bijection from (-∞, ∞) to (lb, ub)
struct ScaledSigmoid{
    T<:Real,
} <: Bijector
    lb::T
    ub::T
end

Bijectors.transform(b::ScaledSigmoid, x::Real) = scaled_sigmoid(x, b.lb, b.ub)
Bijectors.transform(b::ScaledSigmoid, x::AbstractVector{<:Real}) = scaled_sigmoid.(x, b.lb, b.ub)
Bijectors.transform!(b::ScaledSigmoid, x::AbstractVector{<:Real}) = (x .= scaled_sigmoid.(x, b.lb, b.ub))

Bijectors.logabsdetjac(b::ScaledSigmoid, x::Real) = log(abs(d_scaled_sigmoid(x, b.lb, b.ub)))
Bijectors.logabsdetjac(b::ScaledSigmoid, x::AbstractVector{<:Real}) = mapreduce(xi -> d_scaled_sigmoid(xi, b.lb, b.ub), *, x) |> abs |> log

InverseFunctions.inverse(b::ScaledSigmoid) = ScaledLogit(b.lb, b.ub)

# Scaled Logit is a bijection from (lb, ub) to (-∞, ∞)
struct ScaledLogit{
    T<:Real,
} <: Bijector
    lb::T
    ub::T
end

Bijectors.transform(b::ScaledLogit, x::Real) = scaled_logit(x, b.lb, b.ub)
Bijectors.transform(b::ScaledLogit, x::AbstractVector{<:Real}) = scaled_logit.(x, b.lb, b.ub)
Bijectors.transform!(b::ScaledLogit, x::AbstractVector{<:Real}) = (x .= scaled_logit.(x, b.lb, b.ub))

Bijectors.logabsdetjac(b::ScaledLogit, x::Real) = log(abs(d_scaled_logit(x, b.lb, b.ub)))
Bijectors.logabsdetjac(b::ScaledLogit, x::AbstractVector{<:Real}) = mapreduce(xi -> d_scaled_logit(xi, b.lb, b.ub), *, x) |> abs |> log

InverseFunctions.inverse(b::ScaledLogit) = ScaledSigmoid(b.lb, b.ub)
