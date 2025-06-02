
"""
    OptimizationMAP(; kwargs...)

Finds the MAP estimate of the model parameters and hyperparameters using the Optimization.jl package.

To use this model fitter, first add one of the Optimization.jl packages (e.g. OptimizationPRIMA)
to load some optimization algorithms which are passed to the `OptimizationMAP` constructor.

# Keywords
- `algorithm::Any`: Defines the optimization algorithm.
- `multistart::Union{Int, AbstractVector{<:ModelParams}}`: The number of optimization restarts,
        or a vector of `ModelParams` containing initial (hyper)parameter values for the optimization runs.
- `parallel::Bool`: If `parallel=true` then the individual restarts are run in parallel.
- `static_schedule::Bool`: If `static_schedule=true` then the `:static` schedule is used for parallelization.
        This is makes the parallel tasks sticky (non-migrating), but can decrease performance.
- `autodiff::Union{SciMLBase.AbstractADType, Nothing}:`: The automatic differentiation module
    passed to `Optimization.OptimizationFunction`. 
- `kwargs::Base.Pairs{Symbol, <:Any}`: Other kwargs are passed to the optimization algorithm.
"""
struct OptimizationMAP{
    A<:Any,
    S<:Union{Int, AbstractVector{<:ModelParams}},
} <: ModelFitter{MAPParams}
    algorithm::A
    multistart::S
    parallel::Bool
    static_schedule::Bool
    autodiff::SciMLBase.AbstractADType
    kwargs::Base.Pairs{Symbol, <:Any}
end
function OptimizationMAP(;
    algorithm,
    multistart = 200,
    parallel = true,
    static_schedule = false,
    autodiff = AutoForwardDiff(),
    kwargs...
)
    isnothing(autodiff) && (autodiff = SciMLBase.NoAD())
    return OptimizationMAP(algorithm, multistart, parallel, static_schedule, autodiff, kwargs)
end

function set_starts(opt::OptimizationMAP, starts::AbstractVector{<:ModelParams})
    return OptimizationMAP(;
        opt.algorithm,
        multistart = starts,
        opt.parallel,
        opt.static_schedule,
        opt.autodiff,
        opt.kwargs...
    )
end

function slice(opt::OptimizationMAP, idx::Int)
    return OptimizationMAP(;
        opt.algorithm,
        multistart = starts_slice(opt.multistart, idx),
        opt.parallel,
        opt.static_schedule,
        opt.autodiff,
        opt.kwargs...
    )
end

starts_slice(starts::Int, idx::Int) = starts
starts_slice(starts::AbstractVector{<:ModelParams}, idx::Int) = slice.(starts, Ref(idx))

function estimate_parameters(opt::OptimizationMAP, problem::BossProblem, options::BossOptions; return_all::Bool=false)
    if sliceable(problem.model)
        # In case of sliceable model, handle each y dimension separately.
        y_dim_ = y_dim(problem)
        opt_slices = slice.(Ref(opt), 1:y_dim_)
        problem_slices = slice.(Ref(problem), 1:y_dim_)
        
        results = _estimate_parameters.(opt_slices, problem_slices, Ref(options); return_all)
        return reduce_slice_results(results)
    
    else
        # In case of non-sliceable model, optimize all parameters simultaneously.
        return _estimate_parameters(opt, problem, options; return_all)
    end
end

function _estimate_parameters(opt::OptimizationMAP, problem::BossProblem, options::BossOptions; return_all::Bool=false)
    model = problem.model
    data = problem.data

    sampler = params_sampler(model, data)
    params = sampler() # to determine correct parameter shapes in `devectorize`

    vec_, devec_ = vectorizer(model, data)
    bij_ = bijector(model, data)
    inv_bij_ = inverse(bij_)

    vectorize_(params) = bij_(vec_(params))
    devectorize_(ps) = devec_(params, inv_bij_(ps))

    # Prepare the log-likelihood function.
    loglike_ = model_loglike(model, data)
    loglike_vec_ = ps -> loglike_(devectorize_(ps))

    # Skip optimization if there are no free parameters.
    ps = vectorize_(params)
    if length(ps) == 0
        return MAPParams(params, loglike_(params))
    end

    # Generate optimization starts.
    starts = get_starts(opt.multistart, sampler, vectorize_)

    # Optimize.
    ps, loglike = optimize(opt, loglike_vec_, starts, options; return_all)
    
    # Reconstruct the result(s).
    if return_all
        params = devectorize_.(ps)
        return MAPParams.(params, loglike)
    else
        params = devectorize_(ps)
        return MAPParams(params, loglike)
    end
end

function get_starts(multistart::Int, sample_func, vectorize_)
    return reduce(hcat, [vectorize_(sample_func()) for _ in 1:multistart])
end
function get_starts(multistart::AbstractVector{<:ModelParams}, sample_func, vectorize_)
    return reduce(hcat, vectorize_.(multistart))
end

function optimize(
    opt::OptimizationMAP,
    obj,
    starts::AbstractMatrix{<:Real},
    options::BossOptions;
    return_all::Bool = false,
)
    optimization_function = OptimizationFunction((params, _) -> -obj(params), opt.autodiff)
    optimization_problem = (start) -> OptimizationProblem(optimization_function, start, nothing)

    function optim(start)
        params = Optimization.solve(optimization_problem(start), opt.algorithm; opt.kwargs...).u
        ll = obj(params)
        return params, ll
    end

    params, val = optimize_multistart(optim, starts; opt.parallel, opt.static_schedule, options, return_all)
    return params, val
end

# `return_all=false` version
function reduce_slice_results(results::AbstractVector{<:MAPParams})
    params = join_slices(getfield.(results, Ref(:params)))
    loglike = sum(getfield.(results, Ref(:loglike)))
    return MAPParams(params, loglike)
end
# `return_all=true` version
function reduce_slice_results(results::AbstractVector{<:AbstractVector{<:MAPParams}})
    result_matrix = hcat(results...)
    params = (row -> join_slices(getfield.(row, Ref(:params)))).(eachrow(result_matrix))
    loglikes = (row -> sum(getfield.(row, Ref(:loglike)))).(eachrow(result_matrix))
    return MAPParams.(params, loglikes)
end
