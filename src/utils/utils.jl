
"""
    ith(i)(collection) == collection[i]
    ith(i).(collections) == [c[i] for c in collections]
"""
ith(i::Int) = (x) -> x[i]

"""
    cond_func(f)(x, b) == (b ? f(x) : x)
    conf_func(f).(xs, bs) == [b ? f(x) : x for (b,x) in zip(bs,xs)]
"""
cond_func(f::Function) = (x, b) -> b ? f(x) : x

discrete_round(::Nothing, x::AbstractVector{<:Real}) = x
discrete_round(::Missing, x::AbstractVector{<:Real}) = round.(x)
discrete_round(dims::AbstractVector{<:Bool}, x::AbstractVector{<:Real}) = cond_func(round).(x, dims)

"""
    is_feasible(y, y_max) -> Bool

Return true iff `y` satisfies the given constraints.
"""
is_feasible(y::AbstractVector{<:Real}, y_max::AbstractVector{<:Real}) = all(y .<= y_max)

"""
Exclude points exterior to the given `x` domain from the given `X` and `Y` data matrices
and return new matrices `X_` and `Y_`.
"""
function exclude_exterior_points(domain::Domain, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real};
    options::BossOptions=BossOptions(),
)
    interior = in_domain.(eachcol(X), Ref(domain))
    all(interior) && return X, Y
    options.info && @warn "Some data are exterior to the domain and will be discarded!"
    return X[:,interior], Y[:,interior]
end
