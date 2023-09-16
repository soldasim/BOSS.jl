using NLopt

"""
Maximizes the acquisition function using the NLopt.jl library.

Can handle constraints on `x` if according optimization algorithm is selected.

# Fields
  - algorithm: Defines the optimization algorithm.
  - multistart: The number of restarts.
  - parallel: If set to true, the individual optimization runs are run in parallel.
  - kwargs: Stores hyperparameters for the optimization algorithm.
"""
struct NLoptAM <: AcquisitionMaximizer
    algorithm::Symbol
    multistart::Int
    parallel::Bool
    cons_tol::Float64
    kwargs::Base.Pairs{Symbol, <:Any}
end
function NLoptAM(;
    algorithm,
    multistart=200,
    parallel=true,
    cons_tol=1e-8,
    kwargs...
)
    ((:lower_bounds in keys(kwargs)) || (:upper_bounds in keys(kwargs))) && @warn "The provided `:lower_bounds` and `:upper_bounds` kwargs of `BOSS.OptimizationAM` are ignored!\nUse the `domain` field of the `BOSS.OptimizationProblem` instead."
    return NLoptAM(algorithm, multistart, parallel, cons_tol, kwargs)
end

function maximize_acquisition(optimizer::NLoptAM, problem::BOSS.OptimizationProblem, acq::Function; info::Bool)
    domain = problem.domain
    
    if optimizer.multistart == 1
        starts = middle(domain.bounds)[:,:]
    else
        starts = generate_starts_LHC(domain.bounds, optimizer.multistart)
    end

    function acq_optimize(start)
        opt = construct_opt(optimizer, domain, acq, start)
        val, arg, ret = NLopt.optimize(opt, start)
        info && (ret == :FORCED_STOP) && @warn "NLopt optimization terminated with `:FORCED_STOP`!"
        return arg, val
    end
    
    best_x, _ = optimize_multistart(acq_optimize, starts, optimizer.parallel, info)
    return best_x
end

function construct_opt(optimizer::NLoptAM, domain::Domain, acq::Function, start::AbstractVector{<:Real})
    opt = Opt(optimizer.algorithm, x_dim(domain))
    opt.max_objective = (x, g) -> acq(x)
    opt.lower_bounds, opt.upper_bounds = domain.bounds
    add_constraints!(opt, domain.cons, start, optimizer.cons_tol)

    for (s, v) in optimizer.kwargs
        setproperty!(opt, s, v)
    end
    return opt
end

function add_constraints!(opt::Opt, cons::Function, start::AbstractVector{<:Real}, tol::Real)
    c_dim = length(cons(start))
    for i in 1:c_dim
        inequality_constraint!(opt, (x, g) -> -cons(x)[i], tol)
    end
    return opt
end
add_constraints!(opt::Opt, cons::Nothing, start::AbstractVector{<:Real}, tol::Real) = opt
