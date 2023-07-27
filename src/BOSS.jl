module BOSS

export boss!

include("types.jl")
include("parametric.jl")
include("nonparametric.jl")
include("semiparametric.jl")
include("acquisition.jl")
include("algorithms/include.jl")
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
    term_cond::TermCond=IterLimit(1),
    options::BossOptions=BossOptions(),
)
    initialize!(problem)
    while term_cond(problem)
        estimate_parameters!(problem, model_fitter; options)
        x, acq = maximize_acquisition(problem, model_fitter, acq_maximizer; options)
        isnothing(options.plot_options) || make_plot(options.plot_options, problem, acq, x; info=options.info)
        eval_objective!(x, problem; options)
    end
    return problem
end

"""
Perform necessary actions on the input arguments before initiating the optimization.
"""
function initialize!(problem::OptimizationProblem)
    any(problem.discrete) && (problem.model = make_discrete(problem.model, problem.discrete))
end

"""
Estimate the model parameters & hyperparameters using the given `model_fitter` algorithm.
"""
function estimate_parameters!(problem::OptimizationProblem, model_fitter::ModelFitter{T}; options::BossOptions) where {T}
    options.info && @info "Estimating model parameters ..."

    params = estimate_parameters(model_fitter, problem; info=options.info)
    problem.data = update_parameters!(T, problem.data; params...)
end

# Specialized methods of this function for different algorithms are in '\src\algorithms'.
estimate_parameters(model_fitter::ModelFitter, problem::OptimizationProblem; info::Bool) =
    throw(ErrorException("An `estimate_parameters` method for `$(typeof(model_fitter))` does not exist!\nImplement `estimate_parameters(model_fitter::$(typeof(model_fitter)), problem::OptimizationProblem; info::Bool)` method to fix this error."))

"""
Maximize the acquisition via the given `acq_maximizer` algorithm to find the optimal next evaluation point.
"""
function maximize_acquisition(problem::OptimizationProblem, model_fitter::ModelFitter, acq_maximizer::AcquisitionMaximizer; options::BossOptions)
    options.info && @info "Maximizing acquisition function ..."
    
    predict = model_posterior(problem.model, problem.data)
    ϵ_samples = sample_ϵs(y_dim(problem), ϵ_sample_count(model_fitter, options))
    b = best_yet(problem)
    options.info && isnothing(b) && @warn "No feasible solution in the dataset yet. Cannot calculate EI!"

    acq = acquisition(problem.fitness, predict, problem.cons, ϵ_samples, b)
    x = maximize_acquisition(acq_maximizer, problem, acq; info=options.info)
    x = cond_round.(x, problem.discrete)
    return x, acq
end

# Specialized methods of this function for different algorithms are in '\src\algorithms'.
maximize_acquisition(acq_maximizer::AcquisitionMaximizer, problem::OptimizationProblem, acq::Base.Callable; info::Bool) =
    throw(ErrorException("A `maximize_acquisition` method for `$(typeof(acq_maximizer))` does not exist!\nImplement `maximize_acquisition(acq_maximizer::$(typeof(acq_maximizer)), problem::OptimizationProblem, acq::Base.Callable; info::Bool)` method to fix this error."))

function eval_objective!(x::AbstractVector{NUM}, problem::OptimizationProblem{NUM}; options::BossOptions) where {NUM}
    options.info && @info "Evaluating objective function ..."
    
    y = problem.f(x)
    problem.data.X = hcat(problem.data.X, x)
    problem.data.Y = hcat(problem.data.Y, y)

    options.info && @info "New data point: $x : $y"
    return y
end

"""
Return the best fitness among the points in the dataset.
"""
best_yet(problem::OptimizationProblem) = best_yet(problem.fitness, problem.data.Y, problem.cons)
function best_yet(fitness::Fitness, Y::AbstractMatrix{<:Real}, cons::AbstractVector{<:Real})
    isempty(Y) && return nothing
    feasible = is_feasible.(eachcol(Y), Ref(cons))
    any(feasible) || return nothing
    maximum([fitness(Y[:,i]) for i in 1:size(Y)[2] if feasible[i]])
end

"""
Return true iff x belongs to the optimization domain.
"""
function in_domain(domain::AbstractBounds, x::AbstractVector)
    lb, ub = domain
    any(x .< lb) && return false
    any(x .> ub) && return false
    return true
end

"""
Return true iff `y` satisfies the given constraints.
"""
is_feasible(y::AbstractVector{<:Real}, cons::AbstractVector{<:Real}) = all(y .< cons)

"""
Return how many ϵ samples should be drawn. (ϵ represents a random deviation from the model mean.)
"""
ϵ_sample_count(model_fitter::ModelFitter{MLE}, options::BossOptions) = options.ϵ_samples
ϵ_sample_count(model_fitter::ModelFitter{BI}, options::BossOptions) = sample_count(model_fitter)

"""
Round `x` is `b` is true. Otherwise return `x` unchanged. (Useful with broadcasting.)
"""
cond_round(x, b::Bool) = b ? round(x) : x

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
function make_plot(opt::PlotOptions, problem::OptimizationProblem, acquistion::Function, acq_opt::AbstractArray{<:Real}; info::Bool)
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
