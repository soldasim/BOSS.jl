module BOSS

export boss!, result

include("types.jl")
include("utils.jl")
include("optim_utils.jl")
include("domain.jl")
include("models/include.jl")
include("acquisition_function/include.jl")
include("acquisition_maximizer/include.jl")
include("model_fitter/include.jl")
include("term_cond.jl")
include("plot.jl")

"""
    boss!(problem::OptimizationProblem; kwargs...)

Solve the given optimization problem via Bayesian optimization with surrogate model.

# Arguments

- `problem::OptimizationProblem`: Defines the optimization problem.

# Keywords

- `model_fitter::ModelFitter`: Defines the algorithm used to estimate model parameters.
- `acq_maximizer::AcquisitionMaximizer`: Defines the algorithm used to maximize the acquisition function.
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
function boss!(problem::OptimizationProblem;
    model_fitter::ModelFitter=TuringBI(),
    acq_maximizer::AcquisitionMaximizer,
    acquisition::AcquisitionFunction=ExpectedImprovement(),
    term_cond::TermCond=IterLimit(1),
    options::BossOptions=BossOptions(),
)
    initialize!(problem; options)
    while term_cond(problem)
        estimate_parameters!(problem, model_fitter; options)
        x, acq = maximize_acquisition(problem, acquisition, acq_maximizer; options)
        isnothing(options.plot_options) || make_plot(options.plot_options, problem, acq, x; info=options.info)
        eval_objective!(problem, x; options)
    end
    return problem
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
    problem.data = update_parameters!(T, problem.data; params...)
end

"""
Maximize the given `acquisition` function via the given `acq_maximizer` algorithm to find the optimal next evaluation point.
"""
function maximize_acquisition(problem::OptimizationProblem, acquisition::AcquisitionFunction, acq_maximizer::AcquisitionMaximizer; options::BossOptions)
    options.info && @info "Maximizing acquisition function ..."
    acq = acquisition(problem, options)
    x = maximize_acquisition(acq, acq_maximizer, problem, options)
    x = cond_func(round).(problem.domain.discrete, x)
    options.info && !in_domain(x, problem.domain) && @warn "The acquisition maximization resulted in an exterior point $(x)!"
    return x, acq
end

"""
Evaluate the objective function and update the data.
"""
function eval_objective!(problem::OptimizationProblem, x::AbstractVector{<:Real}; options::BossOptions)
    options.info && @info "Evaluating objective function ..."
    
    y = problem.f(x)
    problem.data.X = hcat(problem.data.X, x)
    problem.data.Y = hcat(problem.data.Y, y)

    options.info && @info "New data point: $x : $y"
    return y
end

"""
Estimate the model parameters according to the current data.

For examples of specialized methods see: https://github.com/Sheld5/BOSS.jl/tree/master/src/model_fitter
"""
function estimate_parameters(::ModelFitter, ::OptimizationProblem, ::BossOptions) end

"""
Construct the acquisition function for the given problem.

For examples of specialized methods see: https://github.com/Sheld5/BOSS.jl/tree/master/src/acquisition_function
"""
function (::AcquisitionFunction)(::OptimizationProblem) end

"""
Maximize the given acquisition function.

For examples of specialized methods see: https://github.com/Sheld5/BOSS.jl/tree/master/src/acquisition_maximizer
"""
function maximize_acquisition(::Function, ::AcquisitionMaximizer, ::OptimizationProblem, ::BossOptions) end


"""
Update model parameters.
"""
function update_parameters!(::Type{MLE}, data::ExperimentDataPrior;
    θ=nothing,
    length_scales=nothing,
    noise_vars=nothing,
)
    return ExperimentDataMLE(
        data.X,
        data.Y,
        θ,
        length_scales,
        noise_vars,
    )
end
function update_parameters!(::Type{BI}, data::ExperimentDataPrior;
    θ=nothing,
    length_scales=nothing,
    noise_vars=nothing,
)
    return ExperimentDataBI(
        data.X,
        data.Y,
        θ,
        length_scales,
        noise_vars,
    )
end
function update_parameters!(::Type{T}, data::ExperimentDataPost{T};
    θ=nothing,
    length_scales=nothing,
    noise_vars=nothing,
) where {T<:ModelFit}
    data.θ = θ
    data.length_scales = length_scales
    data.noise_vars = noise_vars
    return data
end
# TODO implement this after type refactor
update_parameters!(::T, problem::OptimizationProblem; kwargs...) where {T<:ModelFit} =
    throw(ArgumentError("Inconsistent experiment data!\nCannot switch from MLE to BI or vice-versa."))

"""
Plot the current state of the optimization process.
"""
function make_plot(opt::PlotOptions, problem::OptimizationProblem, acquistion::Function, acq_opt::AbstractVector{<:Real}; info::Bool)
    info && @info "Plotting ..."
    opt_ = PlotOptions(
        opt.Plots,
        opt.f_true,
        acquistion,
        acq_opt,
        opt.points,
        opt.xaxis,
        opt.yaxis,
        opt.title,
    )
    plot_problem(opt_, problem)
end

end # module BOSS
