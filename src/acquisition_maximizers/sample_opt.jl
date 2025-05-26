
"""
    SampleOptAM(; kwargs...)

Optimizes the acquisition function by first sampling candidates from the user-provided prior,
and then running multiple optimization runs initiated from the samples with the highest acquisition values.

# Keywords
- `x_prior::MultivariateDistribution`: The prior over the input domain used to sample candidates.
- `samples::Int`: The number of samples to be drawn and evaluated.
- `algorithm::Any`: Defines the optimization algorithm.
- `multistart::Int`: The number of optimization restarts.
- `parallel::Bool`: If `parallel=true`, both the sampling and individual optimization runs
        are performed in parallel.
- `autodiff:SciMLBase.AbstractADType:`: The automatic differentiation module
        passed to `Optimization.OptimizationFunction`.
- `kwargs...`: Other kwargs are passed to the optimization algorithm.
"""
struct SampleOptAM{
    A<:Any,
} <: AcquisitionMaximizer
    sampler::SamplingAM
    optimizer::OptimizationAM{A}
end
function SampleOptAM(;
    x_prior,
    samples,
    max_attempts,
    algorithm,
    multistart = 200,
    parallel = true,
    autodiff = AutoForwardDiff(),
    kwargs...
)
    @assert samples >= multistart
    sampler = SamplingAM(x_prior, samples, parallel, max_attempts)
    optimizer = OptimizationAM(algorithm, multistart, parallel, autodiff, kwargs)
    return SampleOptAM(sampler, optimizer)
end

function maximize_acquisition(maximizer::SampleOptAM, problem::BossProblem, options::BossOptions)
    sampler = maximizer.sampler
    opt = maximizer.optimizer
    
    xs, vals = maximize_acquisition(sampler, problem, options; return_all=true)

    xs_score = sortperm(vals) |> reverse
    starts = xs[:,xs_score[1:opt.multistart]]

    x, val = maximize_acquisition(set_starts(opt, starts), problem, options)
    return x, val
end
