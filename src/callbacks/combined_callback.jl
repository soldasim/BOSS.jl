
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
