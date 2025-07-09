
"""
    SampleOptMAP(; kwargs...)
    SampleOptMAP(::SamplingMAP, ::OptimizationMAP)

Combines `SamplingMAP` and `OptimizationMAP` to first sample many parameter samples from the prior,
and subsequently start multiple optimization runs initialized from the best samples.

## Keywords
- `samples::Int`: The number of drawn samples.
- `algorithm::Any`: Defines the optimization algorithm.
- `multistart::Int`: The number of optimization restarts.
- `parallel::Bool`: If `parallel=true`, then both the sampling and the optimization are performed in parallel.
"""
struct SampleOptMAP{
    A<:Any,
} <: ModelFitter{MAPParams}
    sampler::SamplingMAP
    optimizer::OptimizationMAP{A, Int}
end
function SampleOptMAP(;
    samples = 2000,
    algorithm,
    multistart = 200,
    parallel = true,
    autodiff = AutoForwardDiff(),
    kwargs...
)
    isnothing(autodiff) && (autodiff = SciMLBase.NoAD())
    @assert samples >= multistart
    return SampleOptMAP(
        SamplingMAP(samples, parallel),
        OptimizationMAP(algorithm, multistart, parallel, autodiff, kwargs),
    )
end

function estimate_parameters(fitter::SampleOptMAP, problem::BossProblem, options::BossOptions; return_all::Bool=false)
    sampler = fitter.sampler
    opt = fitter.optimizer

    params = estimate_parameters(sampler, problem, options; return_all=true)
    
    sample_score = sortperm(getfield.(params, Ref(:loglike)); rev=true)
    starts = getfield.(params[sample_score[1:opt.multistart]], Ref(:params))

    params = estimate_parameters(set_starts(opt, starts), problem, options; return_all)
    return params
end
