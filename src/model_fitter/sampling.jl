
"""
    SamplingMAP()

Optimizes the model parameters by sampling them from their prior distributions
and selecting the best sample in sense of MAP.

# Keywords
- `samples::Int`: The number of drawn samples.
- `parallel::Bool`: The sampling is performed in parallel if `parallel=true`.
"""
struct SamplingMAP <: ModelFitter{MAP}
    samples::Int
    parallel::Bool
end
SamplingMAP(;
    samples,
    parallel=true,
) = SamplingMAP(samples, parallel)

function estimate_parameters(opt::SamplingMAP, problem::BossProblem, options::BossOptions; return_all::Bool=false)
    loglike = model_loglike(problem.model, problem.noise_std_priors, problem.data)
    sample_() = sample_params(problem.model, problem.noise_std_priors)
    fitness_(p) = loglike(p...)

    params, fit = sample(Val(return_all), Val(opt.parallel), opt, sample_, fitness_)
    return params, fit
end

function sample(return_all::Val{false}, parallel::Val{false}, opt::SamplingMAP, sample_func, fitness_func)
    params, val = sampling_optim(sample_func, fitness_func, opt.samples)
    return params, val
end
function sample(return_all::Val{true}, parallel::Val{false}, opt::SamplingMAP, sample_func, fitness_func)
    samples, vals = sampling_simple(sample_func, fitness_func, opt.samples)
    return samples, vals
end

function sample(return_all::Val{false}, parallel::Val{true}, opt::SamplingMAP, sample_func, fitness_func)
    counts = get_sample_counts(opt.samples, Threads.nthreads())
    ptasks = [Threads.@spawn sampling_optim(sample_func, fitness_func, c) for c in counts]
    results = fetch.(ptasks)

    best = argmax(last.(results))
    return results[best]
end
function sample(return_all::Val{true}, parallel::Val{true}, opt::SamplingMAP, sample_func, fitness_func)
    counts = get_sample_counts(opt.samples, Threads.nthreads())
    ptasks = [Threads.@spawn sampling_simple(sample_func, fitness_func, c) for c in counts]
    results = fetch.(ptasks)

    # `filter` out empty vectors for type stability
    samples = vcat(filter(!isempty, first.(results))...)
    vals = vcat(filter(!isempty, last.(results))...)
    return samples, vals
end

function sampling_optim(sample_func, fitness_func, sample_count::Int)
    best_p = nothing
    best_v = -Inf
    for _ in 1:sample_count
        p = sample_func()
        v = fitness_func(p)
        if v > best_v
            best_p = p
            best_v = v
        end
    end
    return best_p, best_v
end

function sampling_simple(sample_func, fitness_func, sample_count::Int)
    samples = [sample_func() for _ in 1:sample_count]
    vals = fitness_func.(samples)
    return samples, vals
end
