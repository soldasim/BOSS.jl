
"""
    SamplingMAP()

Optimizes the model parameters by sampling them from their prior distributions
and selecting the best sample in sense of MAP.

# Keywords
- `samples::Int`: The number of drawn samples.
- `parallel::Bool`: The sampling is performed in parallel if `parallel=true`.
"""
@kwdef struct SamplingMAP <: ModelFitter{MAPParams}
    samples::Int
    parallel::Bool = true
end

function estimate_parameters(opt::SamplingMAP, problem::BossProblem, options::BossOptions; return_all::Bool=false)
    sampler = params_sampler(problem.model, problem.data)
    loglike_ = model_loglike(problem.model, problem.data)

    params, loglike = sample(Val(return_all), Val(opt.parallel), opt, sampler, loglike_)
    
    if return_all
        return MAPParams.(params, loglike)
    else
        return MAPParams(params, loglike)
    end
end

function sample(return_all::Val{false}, parallel::Val{false}, opt::SamplingMAP, sample_func, loglike)
    params, val = sampling_optim(sample_func, loglike, opt.samples)
    return params, val
end
function sample(return_all::Val{true}, parallel::Val{false}, opt::SamplingMAP, sample_func, loglike)
    samples, vals = sampling_simple(sample_func, loglike, opt.samples)
    return samples, vals
end

function sample(return_all::Val{false}, parallel::Val{true}, opt::SamplingMAP, sample_func, loglike)
    counts = get_sample_counts(opt.samples, Threads.nthreads())
    ptasks = [Threads.@spawn sampling_optim(sample_func, loglike, c) for c in counts]
    results = fetch.(ptasks)

    best = argmax(second.(results))
    return results[best]
end
function sample(return_all::Val{true}, parallel::Val{true}, opt::SamplingMAP, sample_func, loglike)
    counts = get_sample_counts(opt.samples, Threads.nthreads())
    ptasks = [Threads.@spawn sampling_simple(sample_func, loglike, c) for c in counts]
    results = fetch.(ptasks)

    # `filter` out empty vectors for type stability
    samples = vcat(filter(!isempty, first.(results))...)
    vals = vcat(filter(!isempty, second.(results))...)
    return samples, vals
end

# return the best sample only
function sampling_optim(sample_func, loglike, sample_count::Int)
    best_p = nothing
    best_v = -Inf
    for _ in 1:sample_count
        p = sample_func()
        v = loglike(p)
        if v > best_v
            best_p = p
            best_v = v
        end
    end
    return best_p, best_v
end

# return all samples
function sampling_simple(sample_func, loglike, sample_count::Int)
    samples = [sample_func() for _ in 1:sample_count]
    vals = loglike.(samples)
    return samples, vals
end
