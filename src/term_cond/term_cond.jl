
"""
Specifies the termination condition of the whole BOSS algorithm.
Inherit this type to define a custom termination condition.

Example: `struct CustomCond <: TermCond ... end`

All termination conditions *should* implement:
`(cond::CustomCond)(problem::BossProblem)`

This method should return true to keep the optimization running
and return false once the optimization is to be terminated.

See also: [`IterLimit`](@ref)
"""
abstract type TermCond end
