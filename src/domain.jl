
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

in_discrete(x::AbstractVector{<:Real}, discrete::AbstractVector{<:Bool}) =
    all(round.(x[discrete]) .== x[discrete])

in_cons(x::AbstractVector{<:Real}, cons::Nothing) = true
in_cons(x::AbstractVector{<:Real}, cons) = all(cons(x) .>= 0.)

"""
Exclude points exterior to the given `x` domain from the given `X` and `Y` data matrices
and return new matrices `X_` and `Y_`.
"""
function exclude_exterior_points(domain::Domain, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}; options::BossOptions=BossOptions())
    interior = in_domain.(eachcol(X), Ref(domain))
    all(interior) && return X, Y
    options.info && @warn "Some data are exterior to the domain and will be discarded!"
    return X[:,interior], Y[:,interior]
end
