
"""
    OptimizationMAP(; kwargs...)

Finds the MAP estimate of the model parameters and hyperparameters using the Optimization.jl package.

# Keywords
- `algorithm::Any`: Defines the optimization algorithm.
- `multistart::Union{Int, Matrix{Float64}}`: The number of optimization restarts,
        or a vector of tuples `(θ, λ, α)` containing initial (hyper)parameter values for the optimization runs.
- `parallel::Bool`: If `parallel=true` then the individual restarts are run in parallel.
- `softplus_hyperparams::Bool`: If `softplus_hyperparams=true` then the softplus function
        is applied to GP hyperparameters (length-scales & amplitudes) and noise deviations
        to ensure positive values during optimization.
- `softplus_params::Union{Bool, Vector{Bool}}`: Defines to which parameters of the parametric
        model should the softplus function be applied to ensure positive values.
        Supplying a boolean instead of a binary vector turns the softplus on/off for all parameters.
        Defaults to `false` meaning the softplus is applied to no parameters.
"""
struct OptimizationMAP{
    A<:Any,
    S<:Union{<:Int, <:AbstractVector{<:ModelParams}},
} <: ModelFitter{MAP}
    algorithm::A
    multistart::S
    parallel::Bool
    softplus_hyperparams::Bool
    softplus_params::Union{Bool, Vector{Bool}}
    autodiff::SciMLBase.AbstractADType
    kwargs::Base.Pairs{Symbol, <:Any}
end
function OptimizationMAP(;
    algorithm,
    multistart=200,
    parallel=true,
    softplus_hyperparams=true,
    softplus_params=false,
    autodiff=AutoForwardDiff(),
    kwargs...
)
    isnothing(autodiff) && (autodiff = SciMLBase.NoAD())
    return OptimizationMAP(algorithm, multistart, parallel, softplus_hyperparams, softplus_params, autodiff, kwargs)
end

function set_starts(opt::OptimizationMAP, starts::AbstractVector{<:ModelParams})
    return OptimizationMAP(;
        opt.algorithm,
        multistart = starts,
        opt.parallel,
        opt.softplus_hyperparams,
        opt.softplus_params,
        opt.autodiff,
        opt.kwargs...
    )
end

function estimate_parameters(opt::OptimizationMAP, problem::BossProblem, options::BossOptions; return_all::Bool=false)
    # Prepare necessary parameter transformations.
    softplus_mask = create_activation_mask(problem, opt.softplus_hyperparams, opt.softplus_params)
    skip_mask, skipped_values = create_dirac_skip_mask(problem)
    vectorize = (params) -> vectorize_params(params, softplus, softplus_mask, skip_mask)
    devectorize = (params) -> devectorize_params(problem.model, params, softplus, softplus_mask, skipped_values, skip_mask)

    # Skip optimization if there are no parameters.
    if sum(skip_mask) == 0
        return devectorize(Float64[])
    end

    # Generate optimization starts.
    sample_func = () -> sample_params(problem.model, problem.noise_std_priors)
    starts = get_starts(opt.multistart, sample_func, vectorize)

    # Define the optimization objective.
    loglike = model_loglike(problem.model, problem.noise_std_priors, problem.data)
    loglike_vec = (params) -> loglike(devectorize(params)...)

    # Optimize.
    params, loglike = optimize(opt, loglike_vec, starts, options; return_all)
    
    params = return_all ? devectorize.(params) : devectorize(params)
    return params, loglike
end

function get_starts(multistart::Int, sample_func, vectorize)
    return reduce(hcat, [vectorize(sample_func()) for _ in 1:multistart])
end
function get_starts(multistart::AbstractVector{<:ModelParams}, sample_func, vectorize)
    return reduce(hcat, vectorize.(multistart))
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

    function optimize(start)
        params = Optimization.solve(optimization_problem(start), opt.algorithm; opt.kwargs...).u
        ll = obj(params)
        return params, ll
    end

    params, val = optimize_multistart(optimize, starts, opt.parallel, options; return_all)
    return params, val
end
