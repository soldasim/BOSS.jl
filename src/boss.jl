module BOSS

export boss!

include("types.jl")
include("parametric.jl")
include("nonparametric.jl")
include("semiparametric.jl")
include("acquisition.jl")
include("algorithms/include.jl")
include("term_cond.jl")

function boss!(problem::OptimizationProblem;
    model_fitter::ModelFitter=TuringBI(),
    acq_maximizer::AcquisitionMaximizer,
    term_cond::TermCond=IterLimit(1),
    options::BossOptions=BossOptions(),
)
    while term_cond(problem)
        estimate_parameters!(problem, model_fitter; options)
        x = maximize_acquisition(problem, model_fitter, acq_maximizer; options)
        eval_objective!(x, problem; options)
    end
    return problem
end

function estimate_parameters!(problem::OptimizationProblem, model_fitter::ModelFitter{T}; options::BossOptions) where {T}
    options.info && println("Estimating model parameters ...")

    params = estimate_parameters(model_fitter, problem; info=options.info)
    problem.data = update_parameters!(T, problem.data; params...)
end

# Specialized methods of this function are in '\src\algorithms'.
estimate_parameters(model_fitter::ModelFitter, problem::OptimizationProblem; info::Bool) =
    throw(ErrorException("An `estimate_parameters` method for `$(typeof(model_fitter))` does not exist!\nImplement `estimate_parameters(model_fitter::$(typeof(model_fitter)), problem::OptimizationProblem; info::Bool)` method to fix this error."))

function maximize_acquisition(problem::OptimizationProblem, model_fitter::ModelFitter, acq_maximizer::AcquisitionMaximizer; options::BossOptions)
    options.info && println("Maximizing acquisition function ...")
    
    predict = model_posterior(problem.model, problem.data)
    ϵ_samples = sample_ϵs(y_dim(problem), ϵ_sample_count(model_fitter, options))
    b = best_yet(problem)
    options.info && isnothing(b) && @warn "No feasible solution in the dataset yet. Cannot calculate EI!"

    acq = acquisition(problem.fitness, predict, problem.cons, ϵ_samples, b)
    x = maximize_acquisition(acq_maximizer, problem, acq; info=options.info)
    x = cond_round.(x, problem.discrete)
    return x
end

# Specialized methods of this function are in '\src\algorithms'.
maximize_acquisition(acq_maximizer::AcquisitionMaximizer, problem::OptimizationProblem, acq::Base.Callable; info::Bool) =
    throw(ErrorException("A `maximize_acquisition` method for `$(typeof(acq_maximizer))` does not exist!\nImplement `maximize_acquisition(acq_maximizer::$(typeof(acq_maximizer)), problem::OptimizationProblem, acq::Base.Callable; info::Bool)` method to fix this error."))

function eval_objective!(x::AbstractVector{NUM}, problem::OptimizationProblem{NUM}; options::BossOptions) where {NUM}
    options.info && println("Evaluating objective function ...")
    
    y = problem.f(x)
    problem.data.X = hcat(problem.data.X, x)
    problem.data.Y = hcat(problem.data.Y, y)

    options.info && println("New data point: $x : $y")
    return y
end

best_yet(problem::OptimizationProblem) = best_yet(problem.fitness, problem.data.Y, problem.cons)
function best_yet(fitness::Fitness, Y::AbstractMatrix{<:Real}, cons::AbstractVector{<:Real})
    isempty(Y) && return nothing
    feasible = is_feasible.(eachcol(Y), Ref(cons))
    any(feasible) || return nothing
    maximum([fitness(Y[:,i]) for i in 1:size(Y)[2] if feasible[i]])
end

is_feasible(y::AbstractVector{<:Real}, cons::AbstractVector{<:Real}) = all(y .< cons)

ϵ_sample_count(model_fitter::ModelFitter{MLE}, options::BossOptions) = options.ϵ_samples
ϵ_sample_count(model_fitter::ModelFitter{BI}, options::BossOptions) = sample_count(model_fitter)

cond_round(x, b::Bool) = b ? round(x) : x

function update_parameters!(::Type{MLE}, data::ExperimentDataPrior; θ, length_scales, noise_vars)
    return ExperimentDataMLE(
        data.X,
        data.Y,
        θ,
        length_scales,
        noise_vars,
    )
end
function update_parameters!(::Type{BI}, data::ExperimentDataPrior; θ, length_scales, noise_vars)
    return ExperimentDataBI(
        data.X,
        data.Y,
        θ,
        length_scales,
        noise_vars,
    )
end
function update_parameters!(::Type{T}, data::ExperimentDataPost{T}; θ, length_scales, noise_vars) where {T<:ModelFit}
    data.θ = θ
    data.length_scales = length_scales
    data.noise_vars = noise_vars
    return data
end
update_parameters!(::T, problem::OptimizationProblem; kwargs...) where {T<:ModelFit} =
    throw(ArgumentError("Inconsistent experiment data!\nCannot switch from MLE to BI or vice-versa."))

end # module BOSS
