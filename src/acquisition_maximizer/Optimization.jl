using Optimization

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
