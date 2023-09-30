module BOSS

export boss!

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
    boss!(problem)
    boss!(problem; model_fitter, acq_maximizer, term_cond, options)

Solve the given optimization problem via Bayesian optimization with surrogate model.

The optimization problem is defined via the `BOSS.OptimizationProblem` structure
and passed as the sole argument of the function.

The underlying algorithms used for model-fitting and acquisition maximization
can be selected/modified via the `model_fitter` and `acq_maximizer` kwargs.

The termination condition can be modified via the `term_cond` kwarg
and other hyperparameters and options are selected via the `options` kwarg.

See the following docs for more information on the function arguments:
[`BOSS.OptimizationProblem`](@ref),
[`BOSS.ModelFitter`](@ref),
[`BOSS.AcquisitionMaximizer`](@ref),
[`BOSS.TermCond`](@ref),
[`BOSS.Options`](@ref)

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
    initialize!(problem; info=options.info)
    while term_cond(problem)
        estimate_parameters!(problem, model_fitter; options)
        x, acq = maximize_acquisition(problem, acquisition, acq_maximizer; options)
        isnothing(options.plot_options) || make_plot(options.plot_options, problem, acq, x; info=options.info)
        eval_objective!(x, problem; options)
    end
    return problem
end

"""
Perform necessary actions on the input arguments before initiating the optimization.
"""
function initialize!(problem::OptimizationProblem; info::Bool)
    if any(problem.domain.discrete)
        problem.domain = make_discrete(problem.domain)
        problem.model = make_discrete(problem.model, problem.domain.discrete)
    end

    problem.data.X, problem.data.Y = exclude_exterior_points(problem.domain, problem.data.X, problem.data.Y; info)
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

# Specialized methods of this function for different algorithms are in '\src\model_fitter'.
estimate_parameters(model_fitter::ModelFitter, problem::OptimizationProblem, options::BossOptions) =
    throw(ErrorException("An `estimate_parameters` method for `$(typeof(model_fitter))` does not exist!\nImplement `estimate_parameters(model_fitter::$(typeof(model_fitter)), problem::OptimizationProblem; info::Bool)` method to fix this error."))

"""
Maximize the acquisition via the given `acq_maximizer` algorithm to find the optimal next evaluation point.
"""
function maximize_acquisition(problem::OptimizationProblem, acquisition::AcquisitionFunction, acq_maximizer::AcquisitionMaximizer; options::BossOptions)
    options.info && @info "Maximizing acquisition function ..."
    acq = acquisition(problem, options)
    x = maximize_acquisition(acq_maximizer, problem, acq, options)
    x = cond_func(round).(problem.domain.discrete, x)
    return x, acq
end

# Specialized methods of this function for different algorithms are in '\src\acquisition'.
(acquisition::AcquisitionFunction)(problem::OptimizationProblem) =
    throw(ErrorException("Acquisition function for `$(typeof(acquisition))` does not exist!\nImplement `(::$(typeof(acquisition)))(problem::OptimizationProblem)` function to fix this error."))

# Specialized methods of this function for different algorithms are in '\src\acquisition_maximizer'.
maximize_acquisition(acq_maximizer::AcquisitionMaximizer, problem::OptimizationProblem, acq::Function, options::BossOptions) =
    throw(ErrorException("A `maximize_acquisition` method for `$(typeof(acq_maximizer))` does not exist!\nImplement `maximize_acquisition(acq_maximizer::$(typeof(acq_maximizer)), problem::OptimizationProblem, acq::Function; info::Bool)` method to fix this error."))

function eval_objective!(x::AbstractVector{<:Real}, problem::OptimizationProblem; options::BossOptions)
    options.info && @info "Evaluating objective function ..."
    
    y = problem.f(x)
    problem.data.X = hcat(problem.data.X, x)
    problem.data.Y = hcat(problem.data.Y, y)

    options.info && @info "New data point: $x : $y"
    return y
end

"""
Update `θ`,`length_scales`,`noise_vars`, keep `X`,`Y` and return as `BOSS.ExperimentDataPost`.
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
