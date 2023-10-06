
function make_discrete(domain::Domain)
    isnothing(domain.cons) && return domain
    any(domain.discrete) || return domain

    return Domain(
        domain.bounds,
        domain.discrete,
        (x) -> domain.cons(discrete_round(domain.discrete, x)),
    )
end
