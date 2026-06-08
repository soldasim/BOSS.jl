
"""
    NoCallback()

Does nothing.
"""
struct NoCallback <: BossCallback end
(::NoCallback)(::BossProblem; kwargs...) = nothing
