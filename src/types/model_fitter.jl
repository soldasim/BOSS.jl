
"""
    ModelFitter{T<:FittedParams}

Specifies the library/algorithm used for model parameter estimation.
The parametric type `T` specifies the subtype of `FittedParams` returned by the model fitter.

# Defining Custom Model Fitter

Define a custom model fitter algorithm by defining a new subtype of `ModelFitter`.

Example: `struct CustomFitter <: ModelFitter{MAPParams} ... end`

All model fitters *should* implement:
`estimate_parameters(model_fitter::CustomFitter, problem::BossProblem, options::BossOptions; return_all::Bool) -> ::FittedParams

# See Also
[`OptimizationMAP`](@ref), [`TuringBI`](@ref)
"""
abstract type ModelFitter{T<:FittedParams} end
