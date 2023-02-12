using LatinHypercubeSampling

# Activation function to ensure positive optimization arguments.
softplus(x) = log(one(x) + exp(x))

function generate_starts_LHC(bounds::AbstractBounds, count::Int)
    @assert count > 1  # `randomLHC(count, dim)` returns NaNs if `count == 1`
    lb, ub = bounds
    x_dim = length(lb)
    starts = scaleLHC(randomLHC(count, x_dim), [(lb[i], ub[i]) for i in 1:x_dim]) |> transpose
    return starts
end

function random_start(bounds::AbstractBounds)
    lb, ub = bounds
    dim = length(lb)
    start = rand(dim) .* (ub .- lb) .+ lb
    return start
end

function opt_multistart(
    optimize::Base.Callable,  # arg, val = optimize(start)
    starts::AbstractMatrix{<:Real},
    parallel::Bool,
    info::Bool,
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
                if info
                    lock(io_lock)
                    try
                        warn_optim_err(e)
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
                info && warn_optim_err(e)
                errors.value += 1
                args[i] = Float64[]
                vals[i] = -Inf
            end
        end
    end

    (errors.value == multistart) && throw(ErrorException("All acquisition optimization runs failed!"))
    info && (errors.value > 0) && @warn "$(errors.value)/$(multistart) acquisition optimization runs failed!\n"
    b = argmax(vals)
    return args[b], vals[b]
end

function warn_optim_err(e)
    @warn "Optimization error:"
    showerror(stderr, e); println(stderr)
end
