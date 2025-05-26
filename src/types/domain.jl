
"""
    const AbstractBounds = Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}

Defines box constraints.

Example: `([0, 0], [1, 1]) isa AbstractBounds`
"""
const AbstractBounds = Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}

"""
    Domain(; kwargs...)

Describes the optimization domain.

# Keywords
- `bounds::AbstractBounds`: The basic box-constraints on `x`. This field is mandatory.
- `discrete::AbstractVector{Bool}`: Can be used to designate some dimensions
        of the domain as discrete.
- `cons::Union{Nothing, Function}`: Used to define arbitrary nonlinear constraints on `x`.
        Feasible points `x` must satisfy `all(cons(x) .> 0.)`. An appropriate acquisition
        maximizer which can handle nonlinear constraints must be used if `cons` is provided.
        (See [`AcquisitionMaximizer`](@ref).)
"""
@kwdef struct Domain{
    B<:AbstractBounds,
    D<:AbstractVector{Bool},
    C<:Union{Nothing, Function},
}
    bounds::B
    discrete::D = fill(false, length(first(bounds)))
    cons::C = nothing

    function Domain(bounds::B, discrete::D, cons::C) where {B,D,C}
        @assert length(bounds[1]) == length(bounds[2]) == length(discrete)
        return new{B,D,C}(bounds, discrete, cons)
    end
end

x_dim(d::Domain) = length(d.discrete)

cons_dim(d::Domain{<:Any, <:Any, Nothing}) = 0
cons_dim(d::Domain{<:Any, <:Any, <:Any}) = length(d.cons(mean(d.bounds)))

function make_discrete(domain::Domain)
    isnothing(domain.cons) && return domain
    any(domain.discrete) || return domain

    return Domain(
        domain.bounds,
        domain.discrete,
        (x) -> domain.cons(discrete_round(domain.discrete, x)),
    )
end

"""
    in_domain(x, domain) -> Bool

Return true iff x belongs to the domain.
"""
function in_domain(x::AbstractVector{<:Real}, domain::Domain)
    in_bounds(x, domain.bounds) || return false
    in_discrete(x, domain.discrete) || return false
    in_cons(x, domain.cons) || return false
    return true
end

function in_bounds(x::AbstractVector{<:Real}, bounds::AbstractBounds)
    lb, ub = bounds
    any(x .< lb) && return false
    any(x .> ub) && return false
    return true
end

in_discrete(x::AbstractVector{<:Real}, discrete::AbstractVector{Bool}) =
    all(round.(x[discrete]) .== x[discrete])

in_cons(x::AbstractVector{<:Real}, cons::Nothing) = true
in_cons(x::AbstractVector{<:Real}, cons) = all(cons(x) .>= 0.)
