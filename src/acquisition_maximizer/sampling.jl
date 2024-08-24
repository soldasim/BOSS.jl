
"""
    SamplingAM(; kwargs...)

Optimizes the acquisition function by sampling candidates from the user-provided prior,
and returning the sample with the highest acquisition value.

# Keywords
- `x_prior::MultivariateDistribution`: The prior over the input domain used to sample candidates.
- `samples::Int`: The number of samples to be drawn and evaluated.
- `parallel::Bool`: If `parallel=true` then the sampling is parallelized. Defaults to `true`.
"""
struct SamplingAM <: AcquisitionMaximizer
    x_prior::MultivariateDistribution
    samples::Int
    parallel::Bool
end
function SamplingAM(;
    x_prior,
    samples,
    parallel=true,
)
    return SamplingAM(x_prior, samples, parallel)
end

function maximize_acquisition(opt::SamplingAM, acquisition::AcquisitionFunction, problem::BossProblem, options::BossOptions;
    return_all::Bool=false,    
)
    acq = acquisition(problem, options)
    x, val = sample(Val(return_all), Val(opt.parallel), opt.x_prior, opt.samples, acq)
    return x, val
end

function sample(return_all::Val{false}, parallel::Val{false}, x_prior, samples, acq)
    xs = rand(x_prior, samples)
    vals = acq.(eachcol(xs))
    
    best = argmax(vals)
    return xs[:,best], vals[best]
end
function sample(return_all::Val{true}, parallel::Val{false}, x_prior, samples, acq)
    xs = rand(x_prior, samples)
    vals = acq.(eachcol(xs))
    return xs, vals
end

function sample(return_all::Val{false}, parallel::Val{true}, x_prior, samples, acq)
    counts = get_sample_counts(samples, Threads.nthreads())
    ptasks = [Threads.@spawn sample(Val(false), Val(false), x_prior, c, acq) for c in counts]
    results = fetch.(ptasks)
    
    best = argmax(vcat(last.(results)...))
    return results[best]
end
function sample(return_all::Val{true}, parallel::Val{true}, x_prior, samples, acq)
    counts = get_sample_counts(samples, Threads.nthreads())
    ptasks = [Threads.@spawn sample(Val(true), Val(false), x_prior, c, acq) for c in counts]
    results = fetch.(ptasks)
    
    xs = hcat(first.(results)...)
    vals = vcat(last.(results)...)
    return xs, vals
end
