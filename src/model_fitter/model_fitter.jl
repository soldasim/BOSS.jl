
"""
Specifies the library/algorithm used for model parameter estimation.
Inherit this type to define a custom model-fitting algorithms.

Example: `struct CustomFitter <: ModelFitter{MAP} ... end` or `struct CustomFitter <: ModelFitter{BI} ... end`

All model fitters *should* implement:
`estimate_parameters(model_fitter::CustomFitter, problem::BossProblem; info::Bool)`.

This method should return a tuple `(params, val)`.
The returned `params` should be a `ModelParams` (if `CustomAlg <: ModelFitter{MAP}`)
or a `AbstractVector{<:ModelParams}` (if `CustomAlg <: ModelFitter{BI}`).
The returned `val` should be the log likelihood of the parameters (if `CustomAlg <: ModelFitter{MAP}`),
or a vector of log likelihoods of the individual parameter samples (if `CustomAlg <: ModelFitter{BI}`),
or `nothing`.

See also: [`OptimizationMAP`](@ref), [`TuringBI`](@ref)
"""
abstract type ModelFitter{T<:ModelFit} end
