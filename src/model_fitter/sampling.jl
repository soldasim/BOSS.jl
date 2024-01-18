
"""
    SamplingMLE()

Optimizes the model parameters by sampling them from their prior distributions
and selecting the best sample in sense of MLE.
"""
struct SamplingMLE <: ModelFitter{MLE}
    samples::Int
    parallel::Bool
end
SamplingMLE(;
    samples,
    parallel=true,
) = SamplingMLE(samples, parallel)

function estimate_parameters(opt::SamplingMLE, problem::OptimizationProblem, options::BossOptions)
    loglike = model_loglike(problem.model, problem.noise_var_priors, problem.data)
    sample_() = sample_params(problem.model, problem.noise_var_priors)
    fitness_(p) = loglike(p[params_(problem.model)]...)

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

params_(::Parametric) = (:θ, :noise_vars)
params_(::Nonparametric) = (:length_scales, :noise_vars)
params_(::Semiparametric) = (:θ, :length_scales, :noise_vars)

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
