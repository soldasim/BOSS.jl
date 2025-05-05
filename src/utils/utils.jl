
"""
    ith(i)(collection) == collection[i]
    ith(i).(collections) == [c[i] for c in collections]
"""
ith(i::Int) = (x) -> x[i]

second(xs) = xs[2]

"""
    cond_func(f)(x, b) == (b ? f(x) : x)

Useful for combining broadcasting and ternary operator.
"""
cond_func(f::Function) = (x, b::Bool) -> b ? f(x) : x

discrete_round(::Nothing, x::AbstractVector{<:Real}) = x
discrete_round(::Missing, x::AbstractVector{<:Real}) = round.(x)
discrete_round(dims::AbstractVector{Bool}, x::AbstractVector{<:Real}) = cond_func(round).(x, dims)

"""
    is_feasible(y, y_max) -> Bool

Return true iff `y` satisfies the given constraints.
"""
is_feasible(y::AbstractVector{<:Real}, y_max::AbstractVector{<:Real}) = all(y .<= y_max)

"""
    random_point(bounds) -> x

Return a random point form the given bounds.
"""
function random_point(bounds::AbstractBounds)
    lb, ub = bounds
    dim = length(lb)
    start = rand(dim) .* (ub .- lb) .+ lb
    return start
end

"""
    generate_LHC(bounds, count) -> X

Return a matrix of latin hyper-cube vertices in the given bounds.
"""
function generate_LHC(bounds::AbstractBounds, count::Int)
    @assert count > 1  # `randomLHC(count, dim)` returns NaNs if `count == 1`
    lb, ub = bounds
    x_dim = length(lb)
    starts = scaleLHC(randomLHC(count, x_dim), [(lb[i], ub[i]) for i in 1:x_dim]) |> transpose
    return starts
end

# # TODO: unused
# """
# Moves the points to the interior of the given bounds.
# """
# function move_to_interior!(points::AbstractMatrix{<:Float64}, bounds::AbstractBounds; gap=0.)
#     for dim in size(points)[1]
#         points[dim,:] .= move_to_interior.(points[dim,:], Ref((bounds[1][dim], bounds[2][dim])); gap)
#     end
#     return points
# end
# function move_to_interior!(point::AbstractVector{<:Float64}, bounds::AbstractBounds; gap=0.)
#     dim = length(point)
#     point .= move_to_interior.(point, ((bounds[1][i], bounds[2][i]) for i in 1:dim); gap)
# end
# function move_to_interior(point::Float64, bounds::Tuple{<:Float64, <:Float64}; gap=0.)
#     (bounds[2] - point >= gap) || (point = bounds[2] - gap)
#     (point - bounds[1] >= gap) || (point = bounds[1] + gap)
#     return point
# end

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

function ranges(lengths::AbstractVector{<:Integer})
    ranges = UnitRange{eltype(lengths)}[]
    last_i = 0
    for len in lengths
        push!(ranges, (last_i + 1):(last_i + len))
        last_i += len
    end
    return ranges
end
