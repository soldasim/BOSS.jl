using Distributions

"""
The expected improvement (EI) acquisition function.

Measures the quality of a potential evaluation point `x`
as the expected improvement in fitness in comparison to the best-so-far achieved fitness
if `x` was selected as the next evaluation point.

In the case of constrained problem, the expected improvement is additionally weighted by the probability of feasibility.
"""
struct ExpectedImprovement <: AcquisitionFunction end

function (ei::ExpectedImprovement)(problem::OptimizationProblem, options::BossOptions)
    predict = model_posterior(problem.model, problem.data)
    ϵ_samples = sample_ϵs(y_dim(problem), ϵ_sample_count(predict, options))
    b = best_yet(problem)
    options.info && isnothing(b) && @warn "No feasible solution in the dataset yet. Cannot calculate EI!"
    ei(problem.fitness, predict, problem.cons, ϵ_samples, b)
end

(ei::ExpectedImprovement)(fitness::Fitness, posterior::Function, constraints::Nothing, ϵ_samples::AbstractArray{<:Real}, best_yet::Nothing) =
    x -> 0.

(ei::ExpectedImprovement)(fitness::Fitness, posterior::Function, constraints::AbstractVector{<:Real}, ϵ_samples::AbstractArray{<:Real}, best_yet::Nothing) =
    x -> feas_prob(x, posterior, constraints)

(ei::ExpectedImprovement)(fitness::Fitness, posterior::Function, constraints::Nothing, ϵ_samples::AbstractArray{<:Real}, best_yet::Real) =
    x -> expected_improvement(fitness, posterior(x)..., ϵ_samples; best_yet)

(ei::ExpectedImprovement)(fitness::Fitness, posterior::Function, constraints::AbstractVector{<:Real}, ϵ_samples::AbstractArray{<:Real}, best_yet::Real) =
    function acq(x)
        mean, var = posterior(x)
        ei_ = expected_improvement(fitness, mean, var, ϵ_samples; best_yet)
        fp_ = feas_prob(mean, var, constraints)
        return ei_ * fp_
    end

function (ei::ExpectedImprovement)(fitness::Fitness, posteriors::AbstractVector{<:Function}, constraints::AbstractVector{<:Real}, ϵ_samples::AbstractMatrix{<:Real}, best_yet::Union{Nothing, <:Real})
    acqs = ei.(Ref(fitness), posteriors, Ref(constraints), eachcol(ϵ_samples), Ref(best_yet))
    x -> mapreduce(a_ -> a_(x), +, acqs) / length(acqs)
end

feas_prob(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, constraints::Nothing) = 1.
feas_prob(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, constraints::AbstractVector{<:Real}) = prod(cdf.(Distributions.Normal.(mean, var), constraints))

function expected_improvement(fitness::LinFitness, mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, ϵ_samples::AbstractArray{<:Real}; best_yet::Real)
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7352306 Eq(44)
    μf = fitness.coefs' * mean
    σf = sqrt((fitness.coefs .^ 2)' * var)
    norm_ϵ = (μf - best_yet) / σf
    return (μf - best_yet) * cdf(Distributions.Normal(), norm_ϵ) + σf * pdf(Distributions.Normal(), norm_ϵ)
end
function expected_improvement(fitness::NonlinFitness, mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, ϵ_samples::AbstractMatrix{<:Real}; best_yet::Real)
    pred_samples = [mean .+ (var .* ϵ) for ϵ in eachcol(ϵ_samples)]
    return sum(max.(0, fitness.(pred_samples) .- best_yet)) / size(ϵ_samples)[2]
end
function expected_improvement(fitness::NonlinFitness, mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, ϵ::AbstractVector{<:Real}; best_yet::Real)
    pred_sample = mean .+ (var .* ϵ)
    return max(0, fitness(pred_sample) - best_yet)
end

ϵ_sample_count(predict::Function, options::BossOptions) = options.ϵ_samples
ϵ_sample_count(predict::AbstractVector{<:Function}, options::BossOptions) = length(predict)

sample_ϵs(y_dim::Int, sample_count::Int) = rand(Normal(), (y_dim, sample_count))

best_yet(problem::OptimizationProblem) = best_yet(problem.fitness, problem.data.Y, problem.cons)
function best_yet(fitness::Fitness, Y::AbstractMatrix{<:Real}, cons::AbstractVector{<:Real})
    isempty(Y) && return nothing
    feasible = is_feasible.(eachcol(Y), Ref(cons))
    any(feasible) || return nothing
    maximum([fitness(Y[:,i]) for i in 1:size(Y)[2] if feasible[i]])
end
