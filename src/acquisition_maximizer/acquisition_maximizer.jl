
"""
Specifies the library/algorithm used for acquisition function optimization.
Inherit this type to define a custom acquisition maximizer.

Example: `struct CustomAlg <: AcquisitionMaximizer ... end`

All acquisition maximizers *should* implement:
`maximize_acquisition(acq_maximizer::CustomAlg, acq::AcquisitionFunction, problem::BossProblem, options::BossOptions)`.

This method should return a tuple `(x, val)`.
The returned `x` is the point of the input domain which maximizes the given acquisition function `acq` (as a vector),
or a batch of points (as a column-wise matrix).
The returned `val` is the acquisition value `acq(x)`,
or the values `acq.(eachcol(x))` for each point of the batch,
or `nothing` (depending on the acquisition maximizer implementation).

See also: [`OptimizationAM`](@ref)
"""
abstract type AcquisitionMaximizer end
