
"""
    BossOptions(; kwargs...)

Stores miscellaneous settings of the BOSS algorithm.

## Keywords
- `info::Bool`: Setting `info=false` silences the BOSS algorithm.
- `debug::Bool`: Set `debug=true` to print stactraces of caught optimization errors.
- `parallel_evals::Bool`: Defaults to `true`.
        Determines whether to run multiple objective function evaluations
        are parallelized. (Only has an effect if batching AM is used.)
- `callback::BossCallback`: If provided, `callback(::BossProblem; kwargs...)`
        will be called before the BO procedure starts and after every iteration.

See also: [`bo!`](@ref)
"""
struct BossOptions
    info::Bool
    debug::Bool
    parallel_evals::Bool
    callback::BossCallback
end
# @kwdef constructor is in src/deprecated.jl