
"""
    SamplingAM(; kwargs...)

Optimizes the acquisition function by sampling candidates from the user-provided prior,
and returning the sample with the highest acquisition value.

## Keywords
- `x_prior::MultivariateDistribution`: The prior over the input domain used to sample candidates.
- `samples::Int`: The number of samples to be drawn and evaluated.
- `parallel::Bool`: If `parallel=true` then the sampling is parallelized. Defaults to `true`.
"""
@kwdef struct SamplingAM <: AcquisitionMaximizer
    x_prior::MultivariateDistribution
    samples::Int
    parallel::Bool = true
    max_attempts::Int = 200
end

function maximize_acquisition(opt::SamplingAM, problem::BossProblem, options::BossOptions;
    return_all::Bool = false,    
)
    acq = construct_acquisition(problem, options)
    xs, vals = sample(Val(opt.parallel), acq, problem.domain, opt.x_prior, opt.samples; opt.max_attempts)
    
    count = size(xs)[2]
    if count == 0
        @error "SamplingAM: No samples were successfully drawn!\nCheck the `x_prior` and the `Domain`."
    end
    if count < opt.samples
        @warn "SamplingAM: Some samples failed to be drawn!"
    end

    if return_all
        return xs, vals
    else
        best = argmax(vals)
        return xs[:,best], vals[best]
    end
end


function sample(parallel::Val{false}, acq, domain, x_prior, samples; max_attempts::Int)
    xs = [_rand_in_domain(x_prior, domain; max_attempts) for _ in 1:samples]
    xs = _reduce_samples(xs)
    vals = acq.(eachcol(xs))
    return xs, vals
end
function sample(parallel::Val{true}, acq, domain, x_prior, samples; max_attempts::Int)
    counts = get_sample_counts(samples, Threads.nthreads())
    ptasks = [Threads.@spawn sample(Val(false), acq, domain, x_prior, c; max_attempts) for c in counts]
    results = fetch.(ptasks)
    
    xs = hcat(first.(results)...)
    vals = vcat(second.(results)...)
    return xs, vals
end

function _rand_in_domain(x_prior::MultivariateDistribution, domain::Domain; max_attempts::Int=200)
    x = _rand_in_discrete(x_prior, domain.discrete)
    for _ in 1:max_attempts-1
        in_domain(x, domain) && break
        x = _rand_in_discrete(x_prior, domain.discrete)
    end
    if in_domain(x, domain)
        return x
    else
        return nothing
    end
end
function _rand_in_discrete(x_prior::MultivariateDistribution, discrete::AbstractVector{Bool})
    x = rand(x_prior)
    x = cond_func(round).(x, discrete)
    return x
end

function _reduce_samples(xs::AbstractVector{<:Union{Nothing, AbstractVector{<:Real}}})
    return hcat(filter(!isnothing, xs)...)
end
