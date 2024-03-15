
"""
    bo!(problem::OptimizationProblem{Function}; kwargs...)
    x = bo!(problem::OptimizationProblem{Missing}; kwargs...)

Solve the given optimization problem via Bayesian optimization with surrogate model
or give a recommendation for the next evaluation point if `problem.f == missing`.

# Arguments

- `problem::OptimizationProblem`: Defines the optimization problem.

# Keywords

- `model_fitter::ModelFitter`: Defines the algorithm used to estimate model parameters.
- `acq_maximizer::AcquisitionMaximizer`: Defines the algorithm used to maximize the acquisition function.
- `acquisition::AcquisitionFunction`: Defines the acquisition function maximized to select
        promising candidates for further evaluation.
- `term_cond::TermCond`: Defines the termination condition.
- `options::BossOptions`: Defines miscellaneous options and hyperparameters.

# References

[`BOSS.OptimizationProblem`](@ref),
[`BOSS.ModelFitter`](@ref),
[`BOSS.AcquisitionMaximizer`](@ref),
[`BOSS.TermCond`](@ref),
[`BOSS.BossOptions`](@ref)

# Examples
See 'https://github.com/Sheld5/BOSS.jl/tree/master/scripts' for example usage.
"""
function bo!(problem::OptimizationProblem;
    model_fitter::ModelFitter,
    acq_maximizer::AcquisitionMaximizer,
    acquisition::AcquisitionFunction=ExpectedImprovement(),
    term_cond::TermCond=IterLimit(1),
    options::BossOptions=BossOptions(),
)
    initialize!(problem; options)
    while term_cond(problem)
        estimate_parameters!(problem, model_fitter; options)
        X = maximize_acquisition(problem, acquisition, acq_maximizer; options)
        isnothing(options.plot_options) || make_plot(options.plot_options, problem, acquisition, X; info=options.info)
        eval_objective!(problem, X; options)
    end
    return problem
end

function bo!(problem::OptimizationProblem{Missing};
    model_fitter::ModelFitter,
    acq_maximizer::AcquisitionMaximizer,
    acquisition::AcquisitionFunction=ExpectedImprovement(),
    options::BossOptions=BossOptions(),
)
    initialize!(problem; options)
    estimate_parameters!(problem, model_fitter; options)
    X = maximize_acquisition(problem, acquisition, acq_maximizer; options)
    isnothing(options.plot_options) || make_plot(options.plot_options, problem, acquisition, X; info=options.info)
    return X
end

"""
Perform necessary actions and check the problem definition before initiating the optimization.
"""
function initialize!(problem::OptimizationProblem; options::BossOptions)
    if any(problem.domain.discrete)
        problem.domain = make_discrete(problem.domain)
        problem.model = make_discrete(problem.model, problem.domain.discrete)
    end

    problem.data.X, problem.data.Y = exclude_exterior_points(problem.domain, problem.data.X, problem.data.Y; options)
    isempty(problem.data.X) && throw(ErrorException("Cannot start with empty dataset! Provide at least one interior initial point."))

    problem.y_max = [isinf(c) ? Infinity() : c for c in problem.y_max]
end

"""
Estimate the model parameters & hyperparameters using the given `model_fitter` algorithm.
"""
function estimate_parameters!(problem::OptimizationProblem, model_fitter::ModelFitter{T}; options::BossOptions) where {T}
    options.info && @info "Estimating model parameters ..."
    params = estimate_parameters(model_fitter, problem, options)
    problem.data = update_parameters!(T, problem.data, params...)
end

"""
Maximize the given `acquisition` function via the given `acq_maximizer` algorithm to find the optimal next evaluation point.
"""
function maximize_acquisition(problem::OptimizationProblem, acquisition::AcquisitionFunction, acq_maximizer::AcquisitionMaximizer; options::BossOptions)
    options.info && @info "Maximizing acquisition function ..."
    X = maximize_acquisition(acq_maximizer, acquisition, problem, options)
    options.info && check_new_points(X, problem)
    return X
end

function check_new_points(X::AbstractArray{<:Real}, problem::OptimizationProblem)
    for x in eachcol(X)
        in_domain(x, problem.domain) || @warn "The acquisition maximization resulted in an exterior point!\nPoint $(x) not in bounds $(problem.domain.bounds)."
    end
end

"""
Evaluate the objective function and update the data.
"""
function eval_objective!(problem::OptimizationProblem, x::AbstractVector{<:Real}; options::BossOptions)
    options.info && @info "Evaluating objective function ..."
    y = problem.f(x)
    add_data!(problem.data, x, y)
    options.info && @info "New data point: $x : $y"
    return y
end
function eval_objective!(problem::OptimizationProblem, X::AbstractMatrix{<:Real}; options::BossOptions)
    options.info && @info "Evaluating objective function ..."
    if options.parallel_evals
        tasks = [Threads.@spawn problem.f(x) for x in eachcol(X)]
        Y = reduce(hcat, fetch.(tasks))
    else
        Y = mapreduce(problem.f, hcat, eachcol(X))
    end
    add_data!(problem.data, X, Y)
    options.info && @info "New data:" * reduce(*, ("\n\t$x : $y" for (x, y) in zip(eachcol(X), eachcol(Y))))
    return Y
end
