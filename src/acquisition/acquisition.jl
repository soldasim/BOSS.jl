
"""
Specifies the acquisition function describing the "quality" of a potential next evaluation point.
Inherit this type to define a custom acquisition function.

Example: `struct CustomAcq <: AcquisitionFunction ... end`

All acquisition functions *should* implement:
`(acquisition::CustomAcq)(problem::BossProblem, options::BossOptions)`

This method should return a function `acq(x::AbstractVector{<:Real}) = val::Real`,
which is maximized to select the next evaluation function of blackbox function in each iteration.

See also: [`ExpectedImprovement`](@ref)
"""
abstract type AcquisitionFunction end
