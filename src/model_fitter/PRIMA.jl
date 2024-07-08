
"""
    NewuoaMAP(PRIMA; kwargs...)

⚠ The `NewuoaMAP` is deprecated.
Use `OptimizationMAP(Newuoa(), ...)` with `using OptimizationPRIMA` instead.

Finds the MAP of the model parameters and hyperparameters using the NEWUOA algorithm from the PRIMA package.

To use `NewuoaMAP` you need to `] add PRIMA`, evaluate `using PRIMA`
and pass the `PRIMA` module to `NewuoaMAP`.

# Arguments
- `prima::Module`: Provide the `PRIMA` module as it is not a direct dependency of BOSS.

# Keywords
- `multistart::Int`: The number of optimization restarts.
- `parallel::Bool`: If `parallel=true` then the individual restarts are run in parallel.
- `apply_softplus::Bool`: If `apply_softplus=true` then the softplus function is applied
        to GP hyperparameters (length scales & amplitudes) and noise deviations
        to ensure positive values during optimization.
- `softplus_params::Union{Vector{Bool}, Nothing}`: Defines to which parameters of the parametric
        model should the softplus function be applied. Defaults to `nothing` equivalent to all falses.
- Other keywords are passed to the optimization algorithm. See https://github.com/libprima/PRIMA.jl. 
"""
struct NewuoaMAP <: ModelFitter{MAP}
    prima::Module
    multistart::Int
    parallel::Bool
    apply_softplus::Bool
    softplus_params::Union{Vector{Bool}, Nothing}
    kwargs::Base.Pairs{Symbol, <:Any}

    function NewuoaMAP(params...)
        Base.depwarn(
            "The `NewuoaMAP` model fitter is deprecated." *
            "\nUse `OptimizationMAP(Newuoa(), ...)` with `using OptimizationPRIMA` instead.",
            :OptimizationMAP,
        )
        return new(params...)
    end
end
function NewuoaMAP(prima;
    multistart=200,
    parallel=true,
    apply_softplus=true,
    softplus_params=nothing,
    kwargs...
)
    return NewuoaMAP(prima, multistart, parallel, apply_softplus, softplus_params, kwargs)
end

function estimate_parameters(optimizer::NewuoaMAP, problem::BossProblem, options::BossOptions)
    # Prepare necessary parameter transformations.
    softplus_mask = create_activation_mask(problem, optimizer.apply_softplus, optimizer.softplus_params)
    skip_mask, skipped_values = create_dirac_skip_mask(problem)
    vectorize = (params) -> vectorize_params(params..., softplus, softplus_mask, skip_mask)
    devectorize = (params) -> devectorize_params(problem.model, params, softplus, softplus_mask, skipped_values, skip_mask)

    # Generate optimization starts.
    starts = reduce(hcat, (vectorize(sample_params(problem.model, problem.noise_std_priors)) for _ in 1:optimizer.multistart))
    (optimizer.multistart == 1) && (starts = starts[:,:])  # make sure `starts` is a `Matrix`

    # Define the optimization objective.
    loglike = model_loglike(problem.model, problem.noise_std_priors, problem.data)
    loglike_vec = (params) -> loglike(devectorize(params)...)
    
    # Define the objective
    obj = (params) -> -loglike_vec(params)  # `-` beacuse PRIMA.jl minimizes objective

    # Optimize with restarts
    function optimize(start)
        p, info = optimizer.prima.newuoa(obj, start;
            optimizer.kwargs...
        )
        ll = loglike_vec(p)
        return p, ll
    end
    best_params, _ = optimize_multistart(optimize, starts, optimizer.parallel, options)
    return devectorize(best_params)
end
