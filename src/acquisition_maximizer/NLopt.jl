
"""
    NLoptAM(NLopt::Module; kwargs...)

⚠ The `NLoptAM` is deprecated.
Use `OptimizationAM` with `using OptimizationNLopt` instead.

Maximizes the acquisition function using the NLopt.jl library.

Can handle constraints on `x` if according optimization algorithm is selected.

To use `NLoptAM` you need to evaluate `using NLopt` and pass the `NLopt` module to `NLoptAM`.

# Arguments
- `nlopt::Module`: Provide the `NLopt` module as it is not a direct dependency of BOSS.

# Keywords
- `algorithm::Symbol`: Specifies the optimization algorithm from the NLopt algorithms.
- `multistart::Int`: The number of restarts.
- `parallel::Bool`: If set to true, the individual optimization restarts are run in parallel.
- `cons_tol::Float64`: The absolute tolerance of constraint violation.
-  `kwargs...`: Other kwargs are passed to the optimization algorithm.

See also: https://github.com/JuliaOpt/NLopt.jl
"""
struct NLoptAM <: AcquisitionMaximizer
    nlopt::Module
    algorithm::Symbol
    multistart::Int
    parallel::Bool
    cons_tol::Float64
    kwargs::Base.Pairs{Symbol, <:Any}

    function NLoptAM(params...)
        Base.depwarn(
            "The `NLoptAM` acquisition maximizer is deprecated." *
            "\nUse `OptimizationAM` with `using OptimizationNLopt` instead.",
            :OptimizationAM,
        )
        return new(params...)
    end
end
function NLoptAM(nlopt;
    algorithm,
    multistart=200,
    parallel=true,
    cons_tol=1e-18,
    kwargs...
)
    ((:lower_bounds in keys(kwargs)) || (:upper_bounds in keys(kwargs))) && @warn "The provided `:lower_bounds` and `:upper_bounds` kwargs of `OptimizationAM` are ignored!\nUse the `domain` field of the `BossProblem` instead."
    return NLoptAM(nlopt, algorithm, multistart, parallel, cons_tol, kwargs)
end

function maximize_acquisition(optimizer::NLoptAM, acquisition::AcquisitionFunction, problem::BossProblem, options::BossOptions)
    acq = acquisition(problem, options)
    domain = problem.domain
    
    if optimizer.multistart == 1
        starts = mean(domain.bounds)[:,:]
    else
        starts = generate_starts_LHC(domain.bounds, optimizer.multistart)
    end

    function acq_optimize(start)
        opt = construct_opt(optimizer, domain, acq, start)
        _, arg, ret = optimizer.nlopt.optimize(opt, start)
        val = acq(arg)
        options.info && (ret == :FORCED_STOP) && @warn "NLopt optimization terminated with `:FORCED_STOP`!"
        return arg, val
    end
    
    best_x, _ = optimize_multistart(acq_optimize, starts, optimizer.parallel, options)
    best_x = cond_func(round).(problem.domain.discrete, best_x)  # assure discrete dims
    return best_x
end

function construct_opt(optimizer::NLoptAM, domain::Domain, acq::Function, start::AbstractVector{<:Real})
    opt = optimizer.nlopt.Opt(optimizer.algorithm, x_dim(domain))
    opt.lower_bounds, opt.upper_bounds = domain.bounds
    add_objective!(opt, acq)
    add_constraints!(opt, domain.cons, start, optimizer.nlopt, optimizer.cons_tol)
    add_kwargs!(opt, optimizer.kwargs)
    return opt
end

function add_objective!(opt, acq::Function)
    obj = (x) -> -acq(x)  # `obj` is minimized
    function f_obj(x, g)
        if length(g) > 0
            ForwardDiff.gradient!(g, obj, x)
        end
        return obj(x)
    end
    opt.min_objective = f_obj  # `max_objective` is broken with some algorithms.
end

function add_constraints!(opt, cons::Function, start::AbstractVector{<:Real}, nlopt::Module, tol::Real)
    c_dim = length(cons(start))
    cons_neg(x) = (-1.).*cons(x)

    function f_cons(res, x, g)
        if length(g) > 0
            ForwardDiff.jacobian!(g, cons_neg, x)
        end
        res .= cons_neg(x)
    end
    nlopt.inequality_constraint!(opt, f_cons, fill(tol, c_dim))
end
add_constraints!(opt, cons::Nothing, start::AbstractVector{<:Real}, tol::Real) = opt

function add_kwargs!(opt, kwargs::Base.Pairs{Symbol, <:Any})
    for (s, v) in kwargs
        setproperty!(opt, s, v)
    end
end
