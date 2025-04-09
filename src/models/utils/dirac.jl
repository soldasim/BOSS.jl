
"""
    filter_dirac_priors(priors::AbstractVector{<:Distribution}) -> priors_::AbstractVector{<:Union{Distribution, Nothing}}

Return a new vector of priors with all `Dirac` priors removed
(including `Dirac`s embedded in `Product` distributions).
"""
function filter_dirac_priors(priors::AbstractVector{<:Distribution})
    priors_ = _rem_dirac.(priors)
    priors_ = filter(!isnothing, priors_)

    if isempty(priors_)
        return Distribution[]
    else
        return [priors_...]
    end
end

_rem_dirac(prior::Dirac) = nothing
_rem_dirac(prior::Distribution) = prior

function _rem_dirac(prior::Product)
    ds = filter_dirac_priors(prior.v)
    isempty(ds) && return nothing
    return Product(ds)
end

"""
    create_dirac_mask(priors::AbstractVector{<:Union{Distribution, Nothing}) -> (Vector{Bool}, Vector{<:Real})

Create a binary mask to skip all parameters with Dirac priors
from the optimization parameters.

Return the binary skip mask and a vector of the skipped Dirac values.
"""
function create_dirac_mask(priors::AbstractVector{<:Union{Distribution, Nothing}})
    values = get_dirac_values(priors)
    
    is_dirac = .!isnothing.(values)
    dirac_vals = [v for v in values if !isnothing(v)]
    return is_dirac, dirac_vals
end

get_dirac_values(priors) =
    vcat(_get_dirac_value.(priors)...)

_get_dirac_value(prior::Nothing) = nothing
_get_dirac_value(prior::Dirac) = prior.value
_get_dirac_value(prior::UnivariateDistribution) = nothing
_get_dirac_value(prior::MultivariateDistribution) = fill(nothing, length(prior))
_get_dirac_value(prior::Product) = _get_dirac_value.(prior.v)

"""
    filter_diracs(ps::AbstractVector{<:Real}, is_dirac::AbstractVector{Bool}) -> ps_::AbstractVector{<:Real}

Return a new parameter vector with the parameters with Dirac priors skipped.

See also [`insert_diracs`](@ref), which does the opposite.
"""
function filter_diracs(ps::AbstractVector{<:Real}, is_dirac::AbstractVector{Bool})
    ps_ = ps[.!is_dirac]
    return ps_
end

"""
    insert_diracs(ps_::AbstractVector{<:Real}, is_dirac::AbstractVector{Bool}, dirac_vals::AbstractVector{<:Real}) -> ps::AbstractVector{<:Real}

Return a new paremeter vector with the parameters with Dirac priors inserted back.

See also [`filter_diracs`](@ref), which does the opposite.
"""
function insert_diracs(ps_::AbstractVector{<:Real}, is_dirac::AbstractVector{Bool}, dirac_vals::AbstractVector{<:Real})
    ps = similar(ps_, length(is_dirac))
    ps[.!is_dirac] .= ps_
    ps[is_dirac] .= dirac_vals
    return ps
end
