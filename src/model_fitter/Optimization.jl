using Optimization

"""
    OptimizationMLE(; kwargs...)

Finds the MLE of the model parameters and hyperparameters using the Optimization.jl package.

# Keywords
- `algorithm::Any`: Defines the optimization algorithm.
- `multistart::Int`: The number of optimization restarts.
- `parallel::Bool`: If `parallel=true` then the individual restarts are run in parallel.
- `apply_softplus::Bool`: If `apply_softplus=true` then the softplus function is applied
        to noise variances and length scales of GPs to ensure positive values during optimization.
- `softplus_params::Union{Vector{Bool}, Nothing}`: Defines to which parameters of the parametric
        model should the softplus function be applied. Defaults to `nothing` equivalent to all falses.
"""
struct OptimizationMLE{
    A<:Any,
} <: ModelFitter{MLE}
    algorithm::A
    multistart::Int
    parallel::Bool
    apply_softplus::Bool
    softplus_params::Union{Vector{Bool}, Nothing}
    autodiff::SciMLBase.AbstractADType
    kwargs::Base.Pairs{Symbol, <:Any}
end
function OptimizationMLE(;
    algorithm,
    multistart=200,
    parallel=true,
    apply_softplus=true,
    softplus_params=nothing,
    autodiff=AutoForwardDiff(),
    kwargs...
)
    isnothing(autodiff) && (autodiff = SciMLBase.NoAD())
    return OptimizationMLE(algorithm, multistart, parallel, apply_softplus, softplus_params, autodiff, kwargs)
end

function estimate_parameters(opt::OptimizationMLE, problem::BossProblem, options::BossOptions)
    # Prepare necessary parameter transformations.
    softplus_mask = create_activation_mask(problem, opt.apply_softplus, opt.softplus_params)
    skip_mask, skipped_values = create_dirac_skip_mask(problem)
    vectorize = (params) -> vectorize_params(params..., softplus, softplus_mask, skip_mask)
    devectorize = (params) -> devectorize_params(problem.model, params, softplus, softplus_mask, skipped_values, skip_mask)

    # Skip optimization if there are no parameters.
    if sum(skip_mask) == 0
        return devectorize(Float64[])
    end

    # Generate optimization starts.
    starts = reduce(hcat, (vectorize(sample_params(problem.model, problem.noise_var_priors)) for _ in 1:opt.multistart))
    (opt.multistart == 1) && (starts = starts[:,:])  # make sure `starts` is a `Matrix`

    # Define the optimization objective.
    loglike = model_loglike(problem.model, problem.noise_var_priors, problem.data)
    loglike_vec = (params) -> loglike(devectorize(params)...)
    
    # Construct the optimization problem.
    optimization_function = OptimizationFunction((params, _)->(-loglike_vec(params)), opt.autodiff)
    optimization_problem = (start) -> OptimizationProblem(optimization_function, start, nothing)

    # Optimize with restarts.
    function optimize(start)
        params = Optimization.solve(optimization_problem(start), opt.algorithm; opt.kwargs...).u
        ll = loglike_vec(params)
        return params, ll
    end
    best_params, _ = optimize_multistart(optimize, starts, opt.parallel, options)
    return devectorize(best_params)
end
