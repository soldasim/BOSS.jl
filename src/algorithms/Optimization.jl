using Optimization

"""
Stores hyperparameters for the MLE optimization of model parameters.

# Fields
  - algorithm: Defines the optimization algorithm.
  - multistart: The number of restarts.
  - parallel: If set to true, the individual optimization runs are run in parallel.
  - apply_softplus: If set to true, the softplus function is applied to noise variances and length scales of GPs to ensure positive values.
  - softplus_params: Defines to which parameters of the parametric model should the softplus function be applied. Defaults to `nothing` equivalent to all falses.
"""
struct OptimizationMLE{
    A<:Any,
} <: ModelFitter{MLE}
    algorithm::A
    multistart::Int
    parallel::Bool
    apply_softplus::Bool
    softplus_params::Union{Vector{Bool}, Nothing}
    kwargs::Base.Pairs{Symbol, <:Any}
end
function OptimizationMLE(;
    algorithm,
    multistart=200,
    parallel=true,
    apply_softplus=true,
    softplus_params=nothing,
    kwargs...
)
    return OptimizationMLE(algorithm, multistart, parallel, apply_softplus, softplus_params, kwargs)
end

function estimate_parameters(optimization::OptimizationMLE, problem::OptimizationProblem; info::Bool)
    # Prepare necessary parameter transformations.
    softplus_mask = create_activation_mask(problem, optimization.apply_softplus, optimization.softplus_params)
    skip_mask, skipped_values = create_dirac_skip_mask(problem)
    vectorize = (params) -> vectorize_params(params..., softplus, softplus_mask, skip_mask)
    devectorize = (params) -> devectorize_params(problem.model, params, softplus, softplus_mask, skipped_values, skip_mask)

    # Generate optimization starts.
    starts = reduce(hcat, (vectorize(sample_params(problem.model, problem.noise_var_priors)) for _ in 1:optimization.multistart))
    (optimization.multistart == 1) && (starts = starts[:,:])  # make sure `starts` is a `Matrix`

    # Define the optimization objective.
    loglike = model_loglike(problem.model, problem.noise_var_priors, problem.data)
    loglike_vec = (params) -> loglike(devectorize(params)...)
    
    # Construct the optimization problem.
    optimization_function = Optimization.OptimizationFunction((params, _)->(-loglike_vec(params)), AutoForwardDiff())
    optimization_problem = (start) -> Optimization.OptimizationProblem(optimization_function, start, nothing)

    # Optimize with restarts.
    function optimize(start)
        params = Optimization.solve(optimization_problem(start), optimization.algorithm; optimization.kwargs...)
        ll = loglike_vec(params)
        return params, ll
    end
    best_params, _ = optimize_multistart(optimize, starts, optimization.parallel, info)
    return devectorize(best_params)
end

"""
Stores hyperparameters for the acquisition function optimization.

# Fields
  - algorithm: Defines the optimization algorithm.
  - multistart: The number of restarts.
  - parallel: If set to true, the individual optimization runs are run in parallel.
  - kwargs: Stores hyperparameters for the optimization algorithm.
"""
struct OptimizationAM{
    A<:Any,
} <: AcquisitionMaximizer
    algorithm::A
    multistart::Int
    parallel::Bool
    kwargs::Base.Pairs{Symbol, <:Any}
end
function OptimizationAM(;
    algorithm,
    multistart=200,
    parallel=true,
    kwargs...
)
    ((:lb in keys(kwargs)) || (:ub in keys(kwargs))) && @warn "The provided `:lb` and `:ub` kwargs of `BOSS.OptimizationAM` are ignored!\nUse the `domain` field of the `BOSS.OptimizationProblem` instead."
    return OptimizationAM(algorithm, multistart, parallel, kwargs)
end

function maximize_acquisition(optimization::OptimizationAM, problem::BOSS.OptimizationProblem, acq::Function; info::Bool)
    if optimization.multistart == 1
        starts = middle(problem.domain)[:,:]
    else
        starts = generate_starts_LHC(problem.domain, optimization.multistart)
    end

    acq_objective = Optimization.OptimizationFunction((x,p)->acq(x), AutoForwardDiff())
    acq_problem(start) = Optimization.OptimizationProblem(acq_objective, start, nothing; lb=problem.domain[1], ub=problem.domain[2])
    
    function acq_optimize(start)
        x = Optimization.solve(acq_problem(start), optimization.algorithm; optimization.kwargs...)
        a = acq(x)
        return x, a
    end
    
    best_x, _ = optimize_multistart(acq_optimize, starts, optimization.parallel, info)
    return best_x
end
