using Distributions
using Optim
using Turing
using FLoops

# - - - - - - LBFGS OPTIMIZATION - - - - - -

# Find ̂x that maximizes f(x) using parallel multistart LBFGS.
function optim_params(f, starts; kwargs...)
    return optim_params(f, starts, nothing; kwargs...)
end
function optim_params(f, starts, constraints; info=true, debug=false)
    multistart = size(starts)[1]

    results = Vector{Tuple{Vector{Float64}, Float64}}(undef, multistart)
    convergence_errors = 0
    @floop for i in 1:multistart
        try
            results[i] = optim_(f, starts[i,:], constraints)
        catch e
            debug && throw(e)
            @reduce convergence_errors += 1
            results[i] = ([], -Inf)
        end
    end
    info && (convergence_errors > 0) && print("      $(convergence_errors)/$(multistart) optimization runs failed to converge!\n")
    
    opt_i = argmax([res[2] for res in results])
    return results[opt_i]
end

# TODO try changing optimization algorithms
function optim_(f, start, constraints::Nothing)
    opt_res = Optim.optimize(p -> -f(p), start, NelderMead())
    return Optim.minimizer(opt_res), -Optim.minimum(opt_res)
end
function optim_(f, start, constraints::Tuple)
    lb, ub = constraints
    opt_res = Optim.optimize(p -> -f(p), lb, ub, start, Fminbox(LBFGS()))
    return Optim.minimizer(opt_res), -Optim.minimum(opt_res)
end
function optim_(f, start, constraints::TwiceDifferentiableConstraints)
    opt_res = Optim.optimize(p -> -f(p), constraints, start, IPNewton())
    return Optim.minimizer(opt_res), -Optim.minimum(opt_res)
end

# - - - - - - NUTS SAMPLING - - - - - -

"""
Stores hyperparameters of the MC sampler.

Amount of drawn samples:    'chain_count * (warmup + leap_size * sample_count)'
Amount of used samples:     'chain_count * sample_count'

# Fields
  - warmup: The amount of initial discarded samples (in each chain).
  - sample_count: The amount of samples used from each chain.
  - chain_count: The amount of chains sampled.
  - leap_size: The "distance" in a chain between two following used samples.

In each chain;
    Firstly, the first 'warmup' samples are discarded.
    Then additional 'leap_size * sample_count' samples are drawn
    and each 'leap_size'-th of these samples is kept.
Finally, kept samples from all chains are joined and returned.
"""
struct MCSettings
    warmup::Int
    samples_in_chain::Int
    chain_count::Int
    leap_size::Int
end

sample_count(mc::MCSettings) = mc.chain_count * mc.samples_in_chain

# Sample parameters of the given probabilistic model (defined with Turing.jl) using parallel NUTS MC sampling.
function sample_params_nuts(model, param_symbols, mc::MCSettings)
    total_samples = mc.warmup + (mc.leap_size * mc.samples_in_chain)
    chains = Turing.sample(model, NUTS(), MCMCThreads(), total_samples, mc.chain_count)
    samples = [reduce(vcat, chains[s][mc.warmup+mc.leap_size:mc.leap_size:end,:]) for s in param_symbols]
    return samples
end

# - - - - - - DATA GENERATION - - - - - -

# Sample from uniform distribution.
function uniform_sample(a, b, sample_size)
    distr = Product(Distributions.Uniform.(a, b))
    X = rand(distr, sample_size)
    return vec.(collect.(eachslice(X; dims=length(size(X)))))
end

# Sample from log-uniform distribution.
function log_sample(a, b, sample_size)
    X = [exp.(x) for x in uniform_sample(log.(a), log.(b), sample_size)]
    return X
end

# Return points distributed evenly over a given logarithmic range.
function log_range(a, b, len)
    a = log10.(a)
    b = log10.(b)
    range = collect(LinRange(a, b, len))
    range = [10 .^ range[i] for i in 1:len]
    return range
end

# - - - - - - MODEL ERROR - - - - - -

# Calculate RMS error with given y values and models predictions for them.
function rms_error(preds, ys; N=nothing)
    isnothing(N) && (N = length(preds))
    return sqrt((1 / N) * sum((preds .- ys).^2))
end
# Calculate RMS error with given test data.
function rms_error(X, Y, model)
    dims = size(Y)[2]
    preds = reduce(hcat, model.(eachrow(X)))'
    return [rms_error(preds[:,i], Y[:,i]) for i in 1:dims]
end
# Calculate RMS error using uniformly sampled test data.
function rms_error(obj_func, model, a, b, sample_count)
    X = reduce(hcat, uniform_sample(a, b, sample_count))'
    Y = reduce(hcat, obj_func.(X))'
    return rms_error(X, Y, model)
end
