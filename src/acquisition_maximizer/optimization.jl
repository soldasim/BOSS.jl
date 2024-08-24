
"""
    OptimizationAM(; kwargs...)

Maximizes the acquisition function using the Optimization.jl library.

Can handle constraints on `x` if according optimization algorithm is selected.

# Keywords
- `algorithm::Any`: Defines the optimization algorithm.
- `multistart::Union{<:Int, <:AbstractMatrix{<:Real}}`: The number of optimization restarts,
        or a matrix of optimization intial points as columns.
- `parallel::Bool`: If `parallel=true` then the individual restarts are run in parallel.
- `autodiff:SciMLBase.AbstractADType:`: The automatic differentiation module
        passed to `Optimization.OptimizationFunction`.
- `kwargs...`: Other kwargs are passed to the optimization algorithm.
"""
struct OptimizationAM{
    A<:Any,
    S<:Union{<:Int, <:AbstractMatrix{<:Real}},
} <: AcquisitionMaximizer
    algorithm::A
    multistart::S
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
    ((:lb in keys(kwargs)) || (:ub in keys(kwargs))) && @warn "The provided `:lb` and `:ub` kwargs of `OptimizationAM` are ignored!\nUse the `domain` field of the `BossProblem` instead."
    isnothing(autodiff) && (autodiff = SciMLBase.NoAD())
    return OptimizationAM(algorithm, multistart, parallel, autodiff, kwargs)
end

function set_starts(opt::OptimizationAM, starts::AbstractMatrix{<:Real})
    return OptimizationAM(;
        opt.algorithm,
        multistart = starts,
        opt.parallel,
        opt.autodiff,
        opt.kwargs...
    )
end

function maximize_acquisition(opt::OptimizationAM, acquisition::AcquisitionFunction, problem::BossProblem, options::BossOptions)
    domain = problem.domain
    
    acq = acquisition(problem, options)
    cons_func = isnothing(domain.cons) ? nothing : (res, x, p) -> (res .= domain.cons(x))

    starts = get_starts(opt.multistart, domain)

    x, val = optimize(
        opt,
        acq,
        cons_func,
        domain.bounds[1],
        domain.bounds[2],
        domain.discrete,
        cons_dim(domain),
        starts,
        options,
    )
    return x, val
end

function get_starts(multistart::Int, domain::Domain)
    if multistart == 1
        starts = mean(domain.bounds)[:,:]
    else
        starts = generate_starts_LHC(domain.bounds, multistart)
    end
    return starts
end
function get_starts(multistart::AbstractMatrix{<:Real}, domain::Domain)
    return multistart
end

function optimize(
    opt::OptimizationAM,
    obj_func,
    cons_func,
    lb::AbstractVector{<:Real},
    ub::AbstractVector{<:Real},
    discrete::AbstractVector{<:Bool},
    c_dim::Int,
    starts::AbstractMatrix{<:Real},
    options::BossOptions,
)
    acq_objective = OptimizationFunction((x, _) -> -obj_func(x), opt.autodiff; cons=cons_func)
    acq_problem(start) = OptimizationProblem(acq_objective, start, nothing;
        lb,
        ub,
        lcons = fill(0., c_dim),
        ucons = fill(Inf, c_dim),  # Needed for some algs to work.
        int = discrete,
    )

    function acq_optimize(start)
        x = Optimization.solve(acq_problem(start), opt.algorithm; opt.kwargs...).u
        val = obj_func(x)
        return x, val
    end

    best_x, _ = optimize_multistart(acq_optimize, starts, opt.parallel, options)
    best_x = cond_func(round).(discrete, best_x)  # assure discrete dims
    return best_x, obj_func(best_x)
end
