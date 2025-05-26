"""
    bo!(problem::BossProblem{Function}; kwargs...)
    x = bo!(problem::BossProblem{Missing}; kwargs...)

Run the Bayesian optimization procedure to solve the given optimization problem
or give a recommendation for the next evaluation point if `problem.f == missing`.

# Arguments

- `problem::BossProblem`: Defines the optimization problem.

# Keywords

- `model_fitter::ModelFitter`: Defines the algorithm used to estimate model parameters.
- `acq_maximizer::AcquisitionMaximizer`: Defines the algorithm used to maximize the acquisition function.
- `term_cond::TermCond`: Defines the termination condition.
- `options::BossOptions`: Defines miscellaneous settings.

# References

[`BossProblem`](@ref),
[`ModelFitter`](@ref),
[`AcquisitionMaximizer`](@ref),
[`TermCond`](@ref),
[`BossOptions`](@ref)

# Examples
See 'https://soldasim.github.io/BOSS.jl/stable/example/' for example usage.
"""
function bo!(problem::BossProblem;
    model_fitter::ModelFitter,
    acq_maximizer::AcquisitionMaximizer,
    term_cond::TermCond = IterLimit(1),
    options::BossOptions = BossOptions(),
)
    initialize!(problem; options)
    is_consistent(problem) || estimate_parameters!(problem, model_fitter; options)
    options.callback(problem; model_fitter, acq_maximizer, term_cond, options, first=true)
    
    while term_cond(problem)
        X = maximize_acquisition(problem, acq_maximizer; options)
        eval_objective!(problem, X; options)
        estimate_parameters!(problem, model_fitter; options)
        options.callback(problem; model_fitter, acq_maximizer, term_cond, options, first=false)
    end
    
    return problem
end

function bo!(problem::BossProblem{Missing};
    model_fitter::ModelFitter,
    acq_maximizer::AcquisitionMaximizer,
    options::BossOptions = BossOptions(),
)
    initialize!(problem; options)
    is_consistent(problem) || estimate_parameters!(problem, model_fitter; options)
    X = maximize_acquisition(problem, acq_maximizer; options)
    return X
end

"""
Perform some initial sanity checks.
"""
function initialize!(problem::BossProblem; options::BossOptions=BossOptions())
    # problem.data.X, problem.data.Y = exclude_exterior_points(problem.domain, problem.data.X, problem.data.Y; options)
    all_in_domain = in_domain.(eachcol(problem.data.X), Ref(problem.domain)) |> all
    options.info && !all_in_domain && @warn "Some initial datapoints are exterior to the defined domain."

    isempty(problem.data.X) && throw(ErrorException("Cannot start with empty dataset! Provide at least one interior initial point."))
end

"""
    estimate_parameters!(::BossProblem, ::ModelFitter)

Estimate the model parameters & hyperparameters using the given `model_fitter` algorithm.

# Keywords

- `options::BossOptions`: Defines miscellaneous settings.
"""
function estimate_parameters!(problem::BossProblem, model_fitter::ModelFitter{T}; options::BossOptions=BossOptions()) where {T}
    options.info && @info "Estimating model parameters ..."
    params = estimate_parameters(model_fitter, problem, options)
    update_parameters!(problem, params)
end

"""
    x = maximize_acquisition(::BossProblem, ::AcquisitionMaximizer)

Maximize the given `acquisition` function via the given `acq_maximizer` algorithm to find the optimal next evaluation point(s).

# Keywords

- `options::BossOptions`: Defines miscellaneous settings.
"""
function maximize_acquisition(problem::BossProblem, acq_maximizer::AcquisitionMaximizer; options::BossOptions=BossOptions())
    options.info && @info "Maximizing acquisition function ..."
    X, _ = maximize_acquisition(acq_maximizer, problem, options)
    options.info && check_new_points(X, problem)
    return X
end

function check_new_points(X::AbstractArray{<:Real}, problem::BossProblem)
    for x in eachcol(X)
        in_domain(x, problem.domain) || @warn "The acquisition maximization resulted in an exterior point $(x)!"
    end
end

"""
    eval_objective!(::BossProblem, x::AbstractVector{<:Real})

Evaluate the objective function and update the data.

# Keywords

- `options::BossOptions`: Defines miscellaneous settings.
"""
function eval_objective!(problem::BossProblem, x::AbstractVector{<:Real}; options::BossOptions=BossOptions())
    options.info && @info "Evaluating objective function ..."
    y = problem.f(x)
    augment_dataset!(problem, x, y)
    options.info && @info "New data point: $x : $y"
    return y
end
function eval_objective!(problem::BossProblem, X::AbstractMatrix{<:Real}; options::BossOptions=BossOptions())
    options.info && @info "Evaluating objective function ..."
    Y = eval_points(Val(options.parallel_evals), problem.f, X)
    augment_dataset!(problem, X, Y)
    options.info && @info "New data:" * prod(["\n\t$x : $y" for (x, y) in zip(eachcol(X), eachcol(Y))])
    return Y
end

function eval_points(::Val{:serial}, f::Function, X::AbstractMatrix{<:Real})
    Y = mapreduce(f, hcat, eachcol(X))
end
function eval_points(::Val{:parallel}, f::Function, X::AbstractMatrix{<:Real})
    tasks = [Threads.@spawn f(x) for x in eachcol(X)]
    Y = reduce(hcat, fetch.(tasks))
end
function eval_points(::Val{:distributed}, f::Function, X::AbstractMatrix{<:Real})
    Y = @distributed hcat for x in eachcol(X)
        f(x)
    end
end
