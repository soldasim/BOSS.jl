
"""
    AcquisitionFunction

Specifies the acquisition function describing the "quality" of a potential next evaluation point.

# Defining Custom Acquisition Function

To define a custom acquisition function, define a new subtype of `AcquisitionFunction`.
- `struct CustomAcq <: AcquisitionFunction ... end`

All acquisition functions *should* implement:
`(acquisition::CustomAcq)(problem::BossProblem, options::BossOptions) -> (x -> ::Real)`

This method should return a function `acq(x::AbstractVector{<:Real}) = val::Real`,
which is maximized to select the next evaluation function of blackbox function in each iteration.

# See Also
[`ExpectedImprovement`](@ref)
"""
abstract type AcquisitionFunction end
