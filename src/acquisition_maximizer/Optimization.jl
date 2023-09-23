using Optimization
using Zygote

"""
Maximizes the acquisition function using the Optimization.jl library.

Can handle constraints on `x` if according optimization algorithm is selected.

# Fields
  - algorithm: Defines the optimization algorithm.
  - multistart: The number of restarts.
  - parallel: If set to true, the individual optimization runs are run in parallel.
  - autodiff: The automatic differentiation module passed to `Optimization.OptimizationFunction`.
  - kwargs: Stores hyperparameters for the optimization algorithm.
"""
struct OptimizationAM{
    A<:Any,
} <: AcquisitionMaximizer
    algorithm::A
    multistart::Int
    parallel::Bool
    autodiff::SciMLBase.AbstractADType
    kwargs::Base.Pairs{Symbol, <:Any}
end
function OptimizationAM(;
    algorithm,
    multistart=200,
    parallel=true,
    autodiff=AutoForwardDiff(),
    kwargs...
)
    ((:lb in keys(kwargs)) || (:ub in keys(kwargs))) && @warn "The provided `:lb` and `:ub` kwargs of `BOSS.OptimizationAM` are ignored!\nUse the `domain` field of the `BOSS.OptimizationProblem` instead."
    return OptimizationAM(algorithm, multistart, parallel, autodiff, kwargs)
end

function maximize_acquisition(optimizer::OptimizationAM, problem::BOSS.OptimizationProblem, acq::Function, options::BossOptions)
    domain = problem.domain
    
    if optimizer.multistart == 1
        starts = middle(domain.bounds)[:,:]
    else
        starts = generate_starts_LHC(domain.bounds, optimizer.multistart)
    end

    acq_func = (x, p) -> -acq(x)
    cons_func = isnothing(domain.cons) ? nothing : (res, x, p) -> (res .= domain.cons(x))
    
    acq_objective = Optimization.OptimizationFunction(acq_func, optimizer.autodiff; cons=cons_func)
    acq_problem(start) = Optimization.OptimizationProblem(acq_objective, start, nothing;
        lb=domain.bounds[1],
        ub=domain.bounds[2],
        lcons=fill(0., x_dim(problem)),
        int=domain.discrete,
        # `sense` kwarg does not work! (https://github.com/SciML/Optimization.jl/issues/8)
    )

    function acq_optimize(start)
        x = Optimization.solve(acq_problem(start), optimizer.algorithm; optimizer.kwargs...)
        a = acq(x)
        return x, a
    end
    
    best_x, _ = optimize_multistart(acq_optimize, starts, optimizer.parallel, options)
    return best_x
end
