
# - - - GENERATING OPTIMIZATION STARTING POINTS - - - - -

"""
    random_start(bounds) -> x

Return a random point form the given bounds.
"""
function random_start(bounds::AbstractBounds)
    lb, ub = bounds
    dim = length(lb)
    start = rand(dim) .* (ub .- lb) .+ lb
    return start
end

"""
    generate_starts_LHC(bounds, count) -> X

Return a matrix of latin hyper-cube vertices in the given bounds.
"""
function generate_starts_LHC(bounds::AbstractBounds, count::Int)
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
