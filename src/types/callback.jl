
"""
    BossCallback

If a `BossCallback` is provided to `BossOptions`,
the callback is called once before the BO procedure starts,
and after each iteration.

All callbacks *should* implement:
- (::CustomCallback)(::BossProblem;
        ::ModelFitter,
        ::AcquisitionMaximizer,
        ::TermCond,
        ::BossOptions,
        first::Bool,
    )

The kwarg `first` is true only on the first callback before the BO procedure starts.

See `PlotCallback` for an example usage of a callback for plotting.
"""
abstract type BossCallback end

"""
    NoCallback()

Does nothing.
"""
struct NoCallback <: BossCallback end
(::NoCallback)(::BossProblem; kwargs...) = nothing

"""
    CombinedCallback(callbacks...)
    CombinedCallback(::AbstractVector{<:BossCallback})

A callback that combines multiple callbacks and calls all of them.
"""
struct CombinedCallback <: BossCallback
    callbacks::Vector{BossCallback}
end
function CombinedCallback(callbacks::Vararg{BossCallback})
    return CombinedCallback(collect(callbacks))
end

function (cb::CombinedCallback)(problem::BossProblem; kwargs...)
    for c in cb.callbacks
        c(problem; kwargs...)
    end
end
