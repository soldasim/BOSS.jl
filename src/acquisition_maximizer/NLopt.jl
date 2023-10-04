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
    cons_tol=1e-18,
    kwargs...
)
    ((:lower_bounds in keys(kwargs)) || (:upper_bounds in keys(kwargs))) && @warn "The provided `:lower_bounds` and `:upper_bounds` kwargs of `BOSS.OptimizationAM` are ignored!\nUse the `domain` field of the `BOSS.OptimizationProblem` instead."
    return NLoptAM(algorithm, multistart, parallel, cons_tol, kwargs)
end

function maximize_acquisition(acq::Function, optimizer::NLoptAM, problem::BOSS.OptimizationProblem, options::BossOptions)
    domain = problem.domain
    
    if optimizer.multistart == 1
        starts = mean(domain.bounds)[:,:]
    else
        starts = generate_starts_LHC(domain.bounds, optimizer.multistart)
    end

    function acq_optimize(start)
        opt = construct_opt(optimizer, domain, acq, start)
        val, arg, ret = NLopt.optimize(opt, start)
        options.info && (ret == :FORCED_STOP) && @warn "NLopt optimization terminated with `:FORCED_STOP`!"
        return arg, val
    end
    
    best_x, _ = optimize_multistart(acq_optimize, starts, optimizer.parallel, options)
    return best_x
end

function construct_opt(optimizer::NLoptAM, domain::Domain, acq::Function, start::AbstractVector{<:Real})
    opt = Opt(optimizer.algorithm, x_dim(domain))
    opt.lower_bounds, opt.upper_bounds = domain.bounds
    add_objective!(opt, acq)
    add_constraints!(opt, domain.cons, start, optimizer.cons_tol)
    add_kwargs!(opt, optimizer.kwargs)
    return opt
end

function add_objective!(opt::Opt, acq::Function)
    function f_obj(x, g)
        if length(g) > 0
            ForwardDiff.gradient!(g, acq, x)
        end
        return acq(x)
    end
    opt.max_objective = f_obj
end

function add_constraints!(opt::Opt, cons::Function, start::AbstractVector{<:Real}, tol::Real)
    c_dim = length(cons(start))
    cons_neg(x) = (-1.).*cons(x)

    function f_cons(res, x, g)
        if length(g) > 0
            ForwardDiff.jacobian!(g, cons_neg, x)
        end
        res .= cons_neg(x)
    end
    inequality_constraint!(opt, f_cons, fill(tol, c_dim))
end
add_constraints!(opt::Opt, cons::Nothing, start::AbstractVector{<:Real}, tol::Real) = opt

function add_kwargs!(opt::Opt, kwargs::Base.Pairs{Symbol, <:Any})
    for (s, v) in kwargs
        setproperty!(opt, s, v)
    end
end
