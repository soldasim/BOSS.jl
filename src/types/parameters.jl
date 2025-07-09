
"""
    FittedParams{M<:SurrogateModel}

The subtypes of `FittedParams` contain `ModelParams` fitted to the data via different methods.

There are two abstract subtypes of `FittedParams`:
- `UniFittedParams`: Contains a single `ModelParams` instance fitted to the data.
- `MultiFittedParams`: Contains multiple `ModelParams` samples sampled according to the data.

The contained `ModelParams` can be obtained via the `get_params(::FittedParams)` function,
which return either a single `ModelParams` object or a vector of `ModelParams` objects.

All subtypes of `UniFittedParams` implement:
- `get_params(::FittedParams) -> ::ModelParams`
- `slice(::FittedParams, idx::Int) -> ::FittedParams`

All subtypes of `MultiFittedParams` implement:
- `get_params(::FittedParams) -> ::Vector{<:ModelParams}`
- `slice(::FittedParams, idx::Int) -> ::FittedParams`

## See Also

[`MAPParams`](@ref), [`BIParams`](@ref), [`RandomParams`](@ref).
"""
abstract type FittedParams{
    M<:SurrogateModel
} end

"""
    UniFittedParams{M<:SurrogateModel}

An abstract subtype of `FittedParams` that contains a single `ModelParams` instance.

See [`FittedParams`](@ref) for more information.
"""
abstract type UniFittedParams{
    M<:SurrogateModel
} <: FittedParams{M} end

"""
    MultiFittedParams{M<:SurrogateModel}

An abstract subtype of `FittedParams` that contains multiple `ModelParams` samples.

See [`FittedParams`](@ref) for more information.
"""
abstract type MultiFittedParams{
    M<:SurrogateModel
} <: FittedParams{M} end

function get_params end
get_params(params::ModelParams) = params

"""
    FixedParams{M<:SurrogateModel}

Fixed `ModelParams` values for a given `SurrogateModel`.

## Keywords
- `params::ModelParams{M}`: The parameter values.
"""
@kwdef struct FixedParams{
    M<:SurrogateModel,
} <: UniFittedParams{M}
    params::ModelParams{M}
end

get_params(p::FixedParams) = p.params

slice(p::FixedParams, idx::Int) = FixedParams(slice(p.params, idx))

"""
    RandomParams{M<:SurrogateModel}

A single random `ModelParams` sample from the prior.

## Keywords
- `params::ModelParams{M}`: The random model parameters.
"""
@kwdef struct RandomParams{
    M<:SurrogateModel,
} <: UniFittedParams{M}
    params::ModelParams{M}
end

get_params(p::RandomParams) = p.params

slice(p::RandomParams, idx::Int) = RandomParams(slice(p.params, idx))

"""
    MAPParams{M<:SurrogateModel}

`ModelParams` estimated via MAP estimation.

## Keywords
- `params::ModelParams{M}`: The fitted model parameters.
- `loglike::Float64`: The log likelihood of the fitted parameters.
"""
@kwdef struct MAPParams{
    M<:SurrogateModel,
} <: UniFittedParams{M}
    params::ModelParams{M}
    loglike::Union{Float64, Nothing} = nothing
end

get_params(p::MAPParams) = p.params

slice(p::MAPParams, idx::Int) = MAPParams(slice(p.params, idx), nothing)

"""
    BIParams{M<:SurrogateModel, P<:ModelParams{M}}

Contains `ModelParams` samples obtained via (approximate) Bayesian inference.

The individual `ModelParams` samples can be obtained by iterating over the `BIParams` object.

## Keywords
- `samples::Vector{P}`: A vector of the individual model parameter samples.
"""
@kwdef struct BIParams{
    M<:SurrogateModel,
    P<:ModelParams{M},
} <: MultiFittedParams{M}
    samples::Vector{P}
end

get_params(p::BIParams) = p.samples

slice(p::BIParams, idx::Int) = BIParams(slice.(p.samples, Ref(idx)))

Base.getindex(params::BIParams, idx::Int) = params.samples[idx]
Base.length(params::BIParams) = length(params.samples)
Base.eltype(::BIParams{<:Any,P}) where {P} = P

function Base.iterate(params::BIParams)
    if length(params.samples) == 0
        return nothing
    else
        return params.samples[1], 1
    end
end
function Base.iterate(params::BIParams, state::Int)
    state += 1
    if state > length(params.samples)
        return nothing
    else
        return params.samples[state], state
    end
end
