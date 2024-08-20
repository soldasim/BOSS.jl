
"""
    SampleOptMAP(; kwargs...)
    SampleOptMAP(::SamplingMAP, ::OptimizationMAP)

Combines `SamplingMAP` and `OptimizationMAP` to first sample many parameter samples from the prior,
and subsequently start multiple optimization runs initialized from the best samples.

# Keywords
- `samples::Int`: The number of drawn samples.
- `algorithm::Any`: Defines the optimization algorithm.
- `multistart::Int`: The number of optimization restarts.
- `parallel::Bool`: If `parallel=true`, then both the sampling and the optimization are performed in parallel.
- `softplus_hyperparams::Bool`: If `softplus_hyperparams=true` then the softplus function
        is applied to GP hyperparameters (length-scales & amplitudes) and noise deviations
        to ensure positive values during optimization.
- `softplus_params::Union{Bool, Vector{Bool}}`: Defines to which parameters of the parametric
        model should the softplus function be applied to ensure positive values.
        Supplying a boolean instead of a binary vector turns the softplus on/off for all parameters.
        Defaults to `false` meaning the softplus is applied to no parameters.
"""
struct SampleOptMAP{
    A<:Any,
} <: ModelFitter{MAP}
    sampler::SamplingMAP
    optimizer::OptimizationMAP{A, Int}
end
function SampleOptMAP(;
    samples=2000,
    algorithm,
    multistart=200,
    parallel=true,
    softplus_hyperparams=true,
    softplus_params=false,
    autodiff=AutoForwardDiff(),
    kwargs...
)
    isnothing(autodiff) && (autodiff = SciMLBase.NoAD())
    @assert samples >= multistart
    return SampleOptMAP(
        SamplingMAP(samples, parallel),
        OptimizationMAP(algorithm, multistart, parallel, softplus_hyperparams, softplus_params, autodiff, kwargs),
    )
end

function estimate_parameters(fitter::SampleOptMAP, problem::BossProblem, options::BossOptions; return_all::Bool=false)
    sampler = fitter.sampler
    opt = fitter.optimizer

    params, vals = estimate_parameters(sampler, problem, options; return_all=true)
    
    sample_score = sortperm(vals) |> reverse
    starts = params[sample_score[1:opt.multistart]]

    params, vals = estimate_parameters(set_starts(opt, starts), problem, options; return_all)
    return params, vals
end
