using Optimization

# TODO: doc
struct OptimizationMLE{
    A<:Any,
} <: ModelFitter{MLE}
    algorithm::A
    multistart::Int
    parallel::Bool
    softplus_noise_vars::Bool
    kwargs::Base.Pairs{Symbol, <:Any}
end
function OptimizationMLE(;
    algorithm,
    multistart=200,
    parallel=true,
    softplus_noise_vars=true,
    kwargs...
)
    return OptimizationMLE(algorithm, multistart, parallel, softplus_noise_vars, kwargs)
end

function estimate_parameters(optimization::OptimizationMLE, problem::OptimizationProblem; info::Bool)
    starts = reduce(hcat, (sample_params_vec(problem.model, problem.noise_var_priors; softplus_noise_vars=optimization.softplus_noise_vars) for _ in 1:optimization.multistart))
    (optimization.multistart == 1) && (starts = starts[:,:])
    
    loglike = model_loglike(problem.model, problem.noise_var_priors, problem.data)
    loglike_vec(params) = loglike(split_model_params(problem.model, params; softplus_noise_vars=optimization.softplus_noise_vars)...)

    optimization_function = Optimization.OptimizationFunction((params, _)->(-loglike_vec(params)), AutoForwardDiff())
    optimization_problem(start) = Optimization.OptimizationProblem(optimization_function, start, nothing)
    
    function optimize(start)
        params = Optimization.solve(optimization_problem(start), optimization.algorithm; optimization.kwargs...)
        ll = loglike_vec(params)
        return params, ll
    end

    best_params, _ = optimize_multistart(optimize, starts, optimization.parallel, info)
    return split_model_params(problem.model, best_params; softplus_noise_vars=optimization.softplus_noise_vars)
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
