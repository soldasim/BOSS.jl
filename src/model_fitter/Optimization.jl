
"""
    OptimizationMAP(; kwargs...)

Finds the MAP estimate of the model parameters and hyperparameters using the Optimization.jl package.

# Keywords
- `algorithm::Any`: Defines the optimization algorithm.
- `multistart::Int`: The number of optimization restarts.
- `parallel::Bool`: If `parallel=true` then the individual restarts are run in parallel.
- `softplus_hyperparams::Bool`: If `softplus_hyperparams=true` then the softplus function
        is applied to GP hyperparameters (length-scales & amplitudes) and noise deviations
        to ensure positive values during optimization.
- `softplus_params::Union{Bool, Vector{Bool}}`: Defines to which parameters of the parametric
        model should the softplus function be applied to ensure positive values.
        Supplying a boolean instead of a binary vector turns the softplus on/off for all parameters.
        Defaults to `false` meaning the softplus is applied to no parameters.
"""
struct OptimizationMAP{
    A<:Any,
} <: ModelFitter{MAP}
    algorithm::A
    multistart::Int
    parallel::Bool
    softplus_hyperparams::Bool
    softplus_params::Union{Bool, Vector{Bool}}
    autodiff::SciMLBase.AbstractADType
    kwargs::Base.Pairs{Symbol, <:Any}
end
function OptimizationMAP(;
    algorithm,
    multistart=200,
    parallel=true,
    softplus_hyperparams=true,
    softplus_params=false,
    autodiff=AutoForwardDiff(),
    kwargs...
)
    isnothing(autodiff) && (autodiff = SciMLBase.NoAD())
    return OptimizationMAP(algorithm, multistart, parallel, softplus_hyperparams, softplus_params, autodiff, kwargs)
end

function estimate_parameters(opt::OptimizationMAP, problem::BossProblem, options::BossOptions; return_all::Bool=false)
    # Prepare necessary parameter transformations.
    softplus_mask = create_activation_mask(problem, opt.softplus_hyperparams, opt.softplus_params)
    skip_mask, skipped_values = create_dirac_skip_mask(problem)
    vectorize = (params) -> vectorize_params(params..., softplus, softplus_mask, skip_mask)
    devectorize = (params) -> devectorize_params(problem.model, params, softplus, softplus_mask, skipped_values, skip_mask)

    # Skip optimization if there are no parameters.
    if sum(skip_mask) == 0
        return devectorize(Float64[])
    end

    # Generate optimization starts.
    starts = reduce(hcat, (vectorize(sample_params(problem.model, problem.noise_std_priors)) for _ in 1:opt.multistart))
    starts = starts[:,:]  # make sure `starts` is a `Matrix` (relevant when `opt.multistart == 1`)

    # Define the optimization objective.
    loglike = model_loglike(problem.model, problem.noise_std_priors, problem.data)
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
    
    if return_all
        params, vals = optimize_multistart(optimize, starts, opt.parallel, options; return_all=true)
        params = devectorize.(params)
        return params, vals
    else
        best_params, _ = optimize_multistart(optimize, starts, opt.parallel, options)
        return devectorize(best_params)
    end
end
