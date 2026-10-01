
"""
    ParamsCallback()

A [`BossCallback`](@ref) that records the fitted model parameters after each iteration.

`params_history` grows by one entry per callback invocation (once before the BO loop
starts and once after each iteration). Each entry is the [`FittedParams`](@ref) object
stored on the problem at that point in time — its concrete type depends on the model
fitter used (e.g. [`MAPParams`](@ref) for [`OptimizationMAP`](@ref),
[`BIParams`](@ref) for [`TuringBI`](@ref)).

## Example

```julia
cb = ParamsCallback()
options = BossOptions(; callback = cb)
bo!(problem; model_fitter, acq_maximizer, term_cond, options)

# cb.params_history[end]  # FittedParams after the last iteration
```
"""
mutable struct ParamsCallback <: BossCallback
    params_history::Vector{FittedParams}
end

ParamsCallback() = ParamsCallback(FittedParams[])

function (cb::ParamsCallback)(problem::BossProblem; kwargs...)
    isnothing(problem.params) && return
    push!(cb.params_history, problem.params)
end
