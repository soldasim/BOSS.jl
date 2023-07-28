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
    softplus_mask = activation_function_mask(param_count(problem.model) + y_dim(problem), θ_len(problem.model), optimization.apply_softplus, optimization.softplus_params)
    
    # Generate optimization starts.
    starts = reduce(hcat, (sample_params_vec(problem.model, problem.noise_var_priors) for _ in 1:optimization.multistart))
    (optimization.multistart == 1) && (starts = starts[:,:])  # make sure `starts` is a `Matrix`
    if any(softplus_mask)
        for i in 1:optimization.multistart
            starts[:,i] .= cond_func(inv_softplus).(softplus_mask, starts[:,i])
        end
    end

    # Define the optimization objective.
    loglike = model_loglike(problem.model, problem.noise_var_priors, problem.data)
    if any(softplus_mask)
        loglike_vec = (params) -> loglike(split_model_params(problem.model, cond_func(softplus).(softplus_mask, params))...)
    else
        loglike_vec = (params) -> loglike(split_model_params(problem.model, params)...)
    end
    
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
    
    # Return parameters maximizing the likelihood.
    best_params .= cond_func(softplus).(softplus_mask, best_params)
    return split_model_params(problem.model, best_params)
end

function activation_function_mask(
    params_total::Int,
    θ_len::Int,
    mask_noisevar_and_lengthscales::Bool,
    mask_theta::Union{Vector{Bool}, Nothing},
)
    mask = fill(false, params_total)
    if !isnothing(mask_theta)
        mask[1:θ_len] .= mask_theta
    end
    if mask_noisevar_and_lengthscales
        mask[θ_len+1:end] .= true
    end
    return mask
end

# TODO: doc
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

function maximize_acquisition(optimization::OptimizationAM, problem::BOSS.OptimizationProblem, acq::Base.Callable; info::Bool)
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
