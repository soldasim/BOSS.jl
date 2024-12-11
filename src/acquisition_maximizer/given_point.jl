
"""
    GivenPointAM(x::Vector{...})

A dummy acquisition maximizer that always returns predefined point `x`.

# See Also

[`GivenSequenceAM`](@ref),
"""
struct GivenPointAM <: AcquisitionMaximizer
    point::Vector{Float64}
end

"""
    GivenSequenceAM(X::Matrix{...})
    GivenSequenceAM(X::Vector{Vector{...}})

A dummy acquisition maximizer that returns the predefined sequence of points in the given order.
The maximizer throws an error if it runs out of points in the sequence.

# See Also

[`GivenPointAM`](@ref),
"""
mutable struct GivenSequenceAM <: AcquisitionMaximizer
    points::Vector{Vector{Float64}}
    used::Int64
end
GivenSequenceAM(X::AbstractMatrix) = GivenSequenceAM(collect(eachcol(X)))
GivenSequenceAM(X::AbstractVector{<:AbstractVector}) = GivenSequenceAM(X, 0)

function maximize_acquisition(opt::GivenPointAM, ::AcquisitionFunction, ::BossProblem, ::BossOptions)
    return opt.point
end

function maximize_acquisition(opt::GivenSequenceAM, ::AcquisitionFunction, ::BossProblem, ::BossOptions)
    (opt.used == length(opt.points)) && throw(BoundsError("`$(typeof(opt))` run out of predefined evaluation points."))
    opt.used += 1
    return opt.points[opt.used]
end
