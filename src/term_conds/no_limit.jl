
"""
    NoLimit()

Never terminates.
"""
struct NoLimit <: TermCond end
(::NoLimit)(::BossProblem) = true
