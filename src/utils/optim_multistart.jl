
"""
Run the optimization (maximization) procedure contained within the `optimize` argument multiple times
and return the best local optimum found this way.

Return results of all successful runs (not just the best one) if the kwargs `return_all` is set to true.

Throws an error if all optimization runs fail.
"""
function optimize_multistart(
    optimize::Function,  # arg, val = optimize(start)
    starts::AbstractMatrix{<:Real};
    parallel::Bool = true,
    static_schedule::Bool = false, # makes the tasks sticky (non-migrating)
    return_all::Bool = false,
    options::BossOptions = BossOptions(),
)   
    multistart = size(starts)[2]

    args = Vector{Vector{Float64}}(undef, multistart)
    vals = Vector{Float64}(undef, multistart)
    errors = Threads.Atomic{Int}(0)
    
    if parallel
        if static_schedule
            _optim_parallel_static(optimize, multistart, starts, options, args, vals, errors)
        else
            _optim_parallel(optimize, multistart, starts, options, args, vals, errors)
        end
    else
        _optim_serial(optimize, multistart, starts, options, args, vals, errors)
    end

    (errors.value == multistart) && throw(ErrorException("All optimization runs failed!"))
    options.info && (errors.value > 0) && @warn "$(errors.value)/$(multistart) optimization runs failed!\n"
    
    if return_all
        keep = (vals .> -Inf)
        return args[keep], vals[keep]
    else
        b = argmax(vals)
        return args[b], vals[b]
    end
end

function _optim_serial(optimize, multistart, starts, options, args, vals, errors)
    for i in 1:multistart
        try
            a, v = optimize(starts[:,i])
            args[i] = a
            vals[i] = v

        catch e
            warn_optim_err(e, options.info, options.debug)
            errors.value += 1
            args[i] = Float64[]
            vals[i] = -Inf
        end
    end
end
function _optim_parallel(optimize, multistart, starts, options, args, vals, errors)
    Threads.@threads for i in 1:multistart
        try
            a, v = optimize(starts[:,i])
            args[i] = a
            vals[i] = v

        catch e
            warn_optim_err(e, options.info, options.debug)
            Threads.atomic_add!(errors, 1)
            args[i] = Float64[]
            vals[i] = -Inf
        end
    end
end
function _optim_parallel_static(optimize, multistart, starts, options, args, vals, errors)
    Threads.@threads :static for i in 1:multistart
        try
            a, v = optimize(starts[:,i])
            args[i] = a
            vals[i] = v

        catch e
            warn_optim_err(e, options.info, options.debug)
            Threads.atomic_add!(errors, 1)
            args[i] = Float64[]
            vals[i] = -Inf
        end
    end
end

function warn_optim_err(e, info::Bool, debug::Bool)
    if info
        @warn "Optimization error:"
        showerror(stderr, e); println(stderr)
    end
    if debug
        rethrow(e)
    end
end
