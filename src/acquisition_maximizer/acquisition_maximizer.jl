
"""
    AcquisitionMaximizer

Specifies the library/algorithm used for acquisition function optimization.

# Defining Custom Acquisition Maximizer

To define a custom acquisition maximizer, define a new subtype of `AcquisitionMaximizer`.
- `struct CustomAlg <: AcquisitionMaximizer ... end`

All acquisition maximizers *should* implement:
`maximize_acquisition(acq_maximizer::CustomAlg, acq::AcquisitionFunction, problem::BossProblem, options::BossOptions) -> (x, val)`.

This method should return a tuple `(x, val)`. The returned vector `x`
is the point of the input domain which maximizes the given acquisition function `acq` (as a vector),
or a batch of points (as a column-wise matrix).
The returned `val` is the acquisition value `acq(x)`,
or the values `acq.(eachcol(x))` for each point of the batch,
or `nothing` (depending on the acquisition maximizer implementation).

See also: [`OptimizationAM`](@ref)
"""
abstract type AcquisitionMaximizer end
