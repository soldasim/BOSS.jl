# using PRIMA

"""
    CobylaAM(PRIMA::Module; kwargs...)

Maximizes the acquisition function using the new implementation of CobylaAM.

To use `CobylaAM` you need to `] add PRIMA`, evaluate `using PRIMA`
and pass the `PRIMA` module to `CobylaAM`.

Eventually the new algorithm implementations from Prima will be added to Optimization.jl
making `CobylaAM` redundant. (See https://github.com/SciML/Optimization.jl/issues/593.)

# Arguments
- `prima::Module`: Provide the `PRIMA` module as it is not a direct dependency of BOSS.

# Keywords
  - `multistart::Int`: The number of optimization restarts.
  - `parallel::Bool`: If `parallel=true` then the individual restarts are run in parallel.
  - `kwargs...`: Other kwargs are passed to the optimization algorithm. See https://github.com/libprima/PRIMA.jl.
"""
struct CobylaAM <: AcquisitionMaximizer
    prima::Module
    multistart::Int
    parallel::Bool
    kwargs::Base.Pairs{Symbol, <:Any}
end
function CobylaAM(prima;
    multistart=200,
    parallel=true,
    kwargs...
)
    ((:xl in keys(kwargs)) || (:xu in keys(kwargs)) || (:nlconstr in keys(kwargs))) && @warn "The provided `:xl`, `:xu`, `:nlconstr` kwargs of `BOSS.CobylaAM` are ignored!\nUse the `domain` field of the `BOSS.OptimizationProblem` to define the domain instead."
    return CobylaAM(prima, multistart, parallel, kwargs)
end

function maximize_acquisition(optimizer::CobylaAM, acquisition::AcquisitionFunction, problem::OptimizationProblem, options::BossOptions)
    acq = acquisition(problem, options)
    domain = problem.domain
    c_dim = cons_dim(domain)

    if optimizer.multistart == 1
        starts = mean(domain.bounds)[:,:]
    else
        starts = generate_starts_LHC(domain.bounds, optimizer.multistart)
    end

    obj = (x) -> -acq(x)  # `-` beacuse PRIMA.jl minimizes objective
    nonlinear_ineq = isnothing(domain.cons) ? nothing : (x) -> -domain.cons(x)  # `-` because PRIMA.jl defines `nonlinear_ineq(x) ≤ 0`
    xl, xu = domain.bounds

    function acq_optimize(start)
        x, info = optimizer.prima.cobyla(obj, start;
            xl, xu,
            nonlinear_ineq,
            optimizer.kwargs...
        )
        a = acq(x)  # because `fx == -acq(x)`
        return x, a
    end

    best_x, _ = optimize_multistart(acq_optimize, starts, optimizer.parallel, options)
    best_x = cond_func(round).(problem.domain.discrete, best_x)  # assure discrete dims
    return best_x
end
