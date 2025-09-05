
"""
    AcquisitionFunction

Specifies the acquisition function describing the "quality" of a potential next evaluation point.

## Defining custom acquisition function

To define a custom acquisition function, define a new subtype of `AcquisitionFunction`.
- `struct CustomAcq <: AcquisitionFunction ... end`

All acquisition functions *should* implement:
`construct_acquisition(::CustomAcq, ::BossProblem, ::BossOptions) -> (x -> ::Real)`

Acquisition functions *may* implement:
- `get_fitness(::CustomAcq) -> (y -> ::Real)`: Usually will return a callable instance of `Fitness`.

Additionally, the following methods are provided and *need not be implemented*:
- `construct_safe_acquisition(::CustomAcq, ::BossProblem, ::BossOptions) -> (x -> ::Real)`

See the docs of the individual functions for more information.

## See Also

[`construct_acquisition`](@ref),
[`ExpectedImprovement`](@ref)
"""
abstract type AcquisitionFunction end

"""
    construct_acquisition(::AcquisitionFunction, ::BossProblem, ::BossOptions) -> (x -> ::Real)

Construct the given `AcquisitionFunction` for the given `BossProblem`.

The returned function `acq(x::AbstractVector{<:Real}) = val::Real`
is maximized to select the next evaluation function of blackbox function in each iteration.

This method must be implemented for all subtypes of `AcquisitionFunction`.
"""
function construct_acquisition end

"""
    construct_safe_acquisition(::AcquisitionFunction, ::BossProblem, ::BossOptions) -> (x -> ::Real)

Construct the "safe" version of the given `AcquisitionFunction` for the given `BossProblem`.

The safe version returns `-Inf` instead of throwing an error.
"""
function construct_safe_acquisition end

"""
    get_fitness(::AcquisitionFunction) -> (y -> ::Real)

Return the fitness function if the given `AcquisitionFunction` defines it.
Otherwise, throw `MethodError`.
"""
function get_fitness end
