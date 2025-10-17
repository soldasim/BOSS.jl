
"""
    SafeFunction(f, val, info, debug)
    safe_f = make_safe(f, val=nothing; info=true, debug=false)

A wrapper type for a function `f` that may error, returning `val` instead.

Set `info=false` to disable warnings when errors occur.
Set `debug=true` to print the full stacktrace on error.
"""
struct SafeFunction{F,V}
    f::F
    val::V
    info::Bool
    debug::Bool
end

make_safe(f, val=nothing; info=true, debug=false) = SafeFunction(f, val, info, debug)

function (f::SafeFunction)(args...; kwargs...)
    ret = f.val
    try
        ret = f.f(args...; kwargs...)
    catch e
        if f.info
            @warn "Function `$(func_name(f.f))` errored. Returning default value `$(f.val)`."
            Base.showerror(stderr, e)
        end
        if f.debug
            rethrow(e)
        end
    end
    return ret
end

func_name(f::Function) = nameof(f)
func_name(f) = typeof(f).name.name
