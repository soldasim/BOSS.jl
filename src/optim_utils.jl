
"""
Run the optimization (maximization) procedure contained within the `optimize` argument multiple times
and return the best local optimum found this way.
"""
function optimize_multistart(
    optimize::Function,  # arg, val = optimize(start)
    starts::AbstractMatrix{<:Real},
    parallel::Bool,
    options::BossOptions,
)   
    multistart = size(starts)[2]

    args = Vector{Vector{Float64}}(undef, multistart)
    vals = Vector{Float64}(undef, multistart)
    errors = Threads.Atomic{Int}(0)
    
    if parallel
        io_lock = Threads.SpinLock()
        Threads.@threads for i in 1:multistart
            try
                a, v = optimize(starts[:,i])
                args[i] = a
                vals[i] = v

            catch e
                if options.info
                    lock(io_lock)
                    try
                        warn_optim_err(e, options.debug)
                    finally
                        unlock(io_lock)
                    end
                end
                Threads.atomic_add!(errors, 1)
                args[i] = Float64[]
                vals[i] = -Inf
            end
        end

    else
        for i in 1:multistart
            try
                a, v = optimize(starts[:,i])
                args[i] = a
                vals[i] = v

            catch e
                options.info && warn_optim_err(e, options.debug)
                errors.value += 1
                args[i] = Float64[]
                vals[i] = -Inf
            end
        end
    end

    (errors.value == multistart) && throw(ErrorException("All optimization runs failed!"))
    options.info && (errors.value > 0) && @warn "$(errors.value)/$(multistart) optimization runs failed!\n"
    b = argmax(vals)
    return args[b], vals[b]
end

function warn_optim_err(e, debug::Bool)
    @warn "Optimization error:"
    if debug
        showerror(stderr, e, catch_backtrace()); println(stderr)
    else
        showerror(stderr, e); println(stderr)
    end
end
