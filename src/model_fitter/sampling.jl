
"""
    SamplingMAP()

Optimizes the model parameters by sampling them from their prior distributions
and selecting the best sample in sense of MAP.
"""
struct SamplingMAP <: ModelFitter{MAP}
    samples::Int
    parallel::Bool
end
SamplingMAP(;
    samples,
    parallel=true,
) = SamplingMAP(samples, parallel)

function estimate_parameters(opt::SamplingMAP, problem::BossProblem, options::BossOptions)
    loglike = model_loglike(problem.model, problem.noise_std_priors, problem.data)
    sample_() = sample_params(problem.model, problem.noise_std_priors)
    fitness_(p) = loglike(p...)

    if opt.parallel
        counts = get_sample_counts(opt.samples, Threads.nthreads())
        ptasks = [Threads.@spawn sampling_optim(sample_, fitness_, c) for c in counts]
        results = fetch.(ptasks)
        params = [res[1] for res in results]
        vals = [res[2] for res in results]
        b = argmax(vals)
        return params[b]
    else
        params, _ = sampling_optim(sample_, fitness_, opt.samples)
        return params
    end
end

function sampling_optim(sample_func, fitness_func, samples)
    best_p = nothing
    best_v = -Inf
    for _ in 1:samples
        p = sample_func()
        v = fitness_func(p)
        if v > best_v
            best_p = p
            best_v = v
        end
    end
    return best_p, best_v
end

function get_sample_counts(samples, tasks)
    base = floor(samples / tasks) |> Int
    diff = samples - (tasks * base)
    counts = Vector{Int}(undef, tasks)
    counts[1:diff] .= base + 1
    counts[diff+1:end] .= base
    return counts
end
