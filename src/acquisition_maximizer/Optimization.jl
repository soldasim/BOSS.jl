
"""
    OptimizationAM(; kwargs...)

Maximizes the acquisition function using the Optimization.jl library.

Can handle constraints on `x` if according optimization algorithm is selected.

# Keywords
- `algorithm::Any`: Defines the optimization algorithm.
- `multistart::Int`: The number of optimization restarts.
- `parallel::Bool`: If `parallel=true` then the individual restarts are run in parallel.
- `autodiff:SciMLBase.AbstractADType:`: The automatic differentiation module
        passed to `Optimization.OptimizationFunction`.
- `kwargs...`: Other kwargs are passed to the optimization algorithm.
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
    ((:lb in keys(kwargs)) || (:ub in keys(kwargs))) && @warn "The provided `:lb` and `:ub` kwargs of `BOSS.OptimizationAM` are ignored!\nUse the `domain` field of the `BOSS.BossProblem` instead."
    isnothing(autodiff) && (autodiff = SciMLBase.NoAD())
    return OptimizationAM(algorithm, multistart, parallel, autodiff, kwargs)
end

function maximize_acquisition(opt::OptimizationAM, acquisition::AcquisitionFunction, problem::BossProblem, options::BossOptions)
    acq = acquisition(problem, options)
    domain = problem.domain
    c_dim = cons_dim(domain)
    
    if opt.multistart == 1
        starts = mean(domain.bounds)[:,:]
    else
        starts = generate_starts_LHC(domain.bounds, opt.multistart)
    end

    acq_func = (x, p) -> -acq(x)  # `-acq(x)` because Optimization.jl minimizes.
    cons_func = isnothing(domain.cons) ? nothing : (res, x, p) -> (res .= domain.cons(x))
    
    acq_objective = OptimizationFunction(acq_func, opt.autodiff; cons=cons_func)
    acq_problem(start) = OptimizationProblem(acq_objective, start, nothing;
        lb=domain.bounds[1],
        ub=domain.bounds[2],
        lcons=fill(0., c_dim),
        ucons=fill(Inf, c_dim),  # Needed for some algs to work.
        int=domain.discrete,
        # `sense` kwarg does not work! (https://github.com/SciML/Optimization.jl/issues/8)
    )

    function acq_optimize(start)
        x = Optimization.solve(acq_problem(start), opt.algorithm; opt.kwargs...)
        a = acq(x.u)  # because `x.objective == -acq(x.u)`
        return x.u, a
    end
    
    best_x, _ = optimize_multistart(acq_optimize, starts, opt.parallel, options)
    best_x = cond_func(round).(problem.domain.discrete, best_x)  # assure discrete dims
    return best_x
end
