
"""
    LazyFitter(; model_fitter::ModelFitter, data_limit::Union{Nothing, Int}=nothing, data_ratio::Union{Nothing, Float64}=nothing)

A model fitter wrapper that delays parameter refitting based on data accumulation.

This can significantly reduce computational cost as model parameter fitting
can often be the most expensive step in the optimization loop.

The fitter supports two modes (exactly one must be specified):
- **data_limit**: Refits once at least `data_limit` new data points have been added since the last fitting.
- **data_ratio**: Refits once the ratio `new_data_size / old_data_size >= data_ratio`.
  For example, `data_ratio=2.0` will refit once the dataset has doubled.

## Keywords
- `model_fitter::ModelFitter`: The underlying model fitter to call periodically.
- `data_limit::Union{Nothing, Int}`: The minimum number of new data points required to trigger a refit. Must be positive if specified.
- `data_ratio::Union{Nothing, Float64}`: The data size ratio threshold for triggering a refit. Must be > 1.0 if specified.

## Examples
```julia
# Refit every 10 data points
fitter1 = LazyFitter(model_fitter=OptimizationMAP(...), data_limit=10)

# Refit once dataset doubles
fitter2 = LazyFitter(model_fitter=OptimizationMAP(...), data_ratio=2.0)
```
"""
@kwdef struct LazyFitter{
    P<:FittedParams,
    M<:ModelFitter{P},
} <: ModelFitter{P}
    model_fitter::M
    data_limit::Union{Nothing, Int} = nothing
    data_ratio::Union{Nothing, Float64} = nothing
    # Track the last data size; uses a Ref for mutable state across function calls
    last_data_size::Ref{Int} = Ref(0)
    
    function LazyFitter(model_fitter::M, data_limit, data_ratio, last_data_size) where {
        P<:FittedParams,
        M<:ModelFitter{P},
    }
        @assert xor(isnothing(data_limit), isnothing(data_ratio)) "Exactly one of data_limit or data_ratio keywords must be defined."
        if !isnothing(data_limit) && data_limit <= 0
            throw(ArgumentError("LazyFitter: data_limit must be positive, got $data_limit"))
        end
        if !isnothing(data_ratio) && data_ratio <= 1.0
            throw(ArgumentError("LazyFitter: data_ratio must be > 1.0, got $data_ratio"))
        end
        
        return new{P, M}(
            model_fitter, data_limit, data_ratio, last_data_size
        )
    end
end

function estimate_parameters(fitter::LazyFitter, problem::BossProblem, options::BossOptions; return_all::Bool=false)
    current_data_size = length(problem.data)
    last_size = fitter.last_data_size[]
    
    # Determine if we should refit based on the mode
    should_refit = false
    
    if last_size == 0
        # Always refit on first call
        should_refit = true
    elseif !isnothing(fitter.data_limit)
        should_refit = (current_data_size - last_size >= fitter.data_limit)
    elseif !isnothing(fitter.data_ratio)
        should_refit = (current_data_size >= fitter.data_ratio * last_size)
    else
        error() # should never happen due to constructor assertion
    end
    
    if should_refit
        params = estimate_parameters(fitter.model_fitter, problem, options; return_all)
        fitter.last_data_size[] = current_data_size
        return params
    end
    
    # If no parameters exist yet (shouldn't happen in normal usage), fall back to fitting
    if isnothing(problem.params)
        @warn "LazyFitter: No existing parameters found, refitting..."
        params = estimate_parameters(fitter.model_fitter, problem, options; return_all)
        fitter.last_data_size[] = current_data_size
        return params
    end
    
    # Not enough data added; reuse existing parameters
    return problem.params
end
