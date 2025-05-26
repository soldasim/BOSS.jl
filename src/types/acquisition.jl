
"""
    AcquisitionFunction

Specifies the acquisition function describing the "quality" of a potential next evaluation point.

# Defining Custom Acquisition Function

To define a custom acquisition function, define a new subtype of `AcquisitionFunction`.
- `struct CustomAcq <: AcquisitionFunction ... end`

All acquisition functions *should* implement:
`construct_acquisition(::CustomAcq, ::BossProblem, ::BossOptions) -> (x -> ::Real)`

Acquisition functions *may* implement:
- `get_fitness(::CustomAcq) -> (y -> ::Real)`: Usually will return a callable instance of `Fitness`.

This method should return a function `acq(x::AbstractVector{<:Real}) = val::Real`,
which is maximized to select the next evaluation function of blackbox function in each iteration.

# See Also
[`ExpectedImprovement`](@ref)
"""
abstract type AcquisitionFunction end

"""
    construct_acquisition(::AcquisitionFunction, ::BossProblem, ::BossOptions) -> (x -> ::Real)

Construct the given `AcquisitionFunction` for the given `BossProblem`.

This method must be implemented for all subtypes of `AcquisitionFunction`.
"""
function construct_acquisition end

"""
    get_fitness(::AcquisitionFunction) -> (y -> ::Real)

Return the fitness function if the given `AcquisitionFunction` defines it.
Otherwise, throw `MethodError`.
"""
function get_fitness end
