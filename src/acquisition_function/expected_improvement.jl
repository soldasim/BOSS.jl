using Distributions

# TODO: remove unnecessary ϵ sampling for `LinFitness`

"""
    ExpectedImprovement(; kwargs...)

The expected improvement (EI) acquisition function.

Measures the quality of a potential evaluation point `x` as the expected improvement
in best-so-far achieved fitness by evaluating the objective function at `y = f(x)`.

In case of constrained problems, the expected improvement is additionally weighted
by the probability of feasibility of `y`. I.e. the probability that `all(cons(y) .> 0.)`.

If the problem is constrained on `y` and no feasible point has been observed yet,
then the probability of feasibility alone is returned as the acquisition function.

Rather than using the actual evaluations `(xᵢ,yᵢ)` from the dataset,
the best-so-far achieved fitness is calculated as the maximum fitness
among the means `̂yᵢ` of the posterior predictive distribution of the model
evaluated at `xᵢ`. This is a simple way to handle evaluation noise which may not
be suitable for problems with substantial noise. In case of Bayesian Inference,
an averaged posterior of the model posterior samples is used for the prediction of `ŷᵢ`.

# Keywords
- `ϵ_samples::Int`: Controls how many samples are used to approximate EI.
        The `ϵ_samples` keyword is *ignored* unless `MLE` model fitter and `NonlinFitness` are used!
        In case of `BI` model fitter, the number of samples is instead set equal to the number of posterior samples.
        In case of `LinearFitness`, the expected improvement can be calculated analytically.

- `cons_safe::Bool`: If set to true, the acquisition function `acq(x)` is made 'constraint-safe'
        by wrapping it in `safe_acq(x) = in_domain(x, domain) ? acq(x) : 0.`.
        Set `cons_safe` to `true` if the evaluation of the model at exterior points
        may cause errors or is nonsensical.
        Set `cons_safe` to `false` if the evaluation of the model at exterior points
        can provide information to the acquisition maximizer.
        Defaults to `false`.
"""
struct ExpectedImprovement <: AcquisitionFunction
    ϵ_samples::Int  # only used in case of MLE and NonlinFitness
    cons_safe::Bool
end
function ExpectedImprovement(;
    ϵ_samples=200,
    cons_safe=false,
)
    return ExpectedImprovement(ϵ_samples, cons_safe)
end

function (ei::ExpectedImprovement)(problem::OptimizationProblem, options::BossOptions)
    posterior = model_posterior(problem.model, problem.data)
    ϵ_samples = sample_ϵs(y_dim(problem), ϵ_sample_count(posterior, ei.ϵ_samples))
    b = best_so_far(problem, posterior)
    options.info && isnothing(b) && @warn "No feasible solution in the dataset yet. Cannot calculate EI!"
    acq = ei(problem.fitness, posterior, problem.y_max, ϵ_samples, b)
    return ei.cons_safe ? make_safe(acq, problem.domain) : acq
end

function make_safe(acq::Function, domain::Domain)
    return acq_safe(x) = in_domain(x, domain) ? acq(x) : 0.
end

# Construct acquisition function from a single posterior.
(ei::ExpectedImprovement)(fitness::Fitness, posterior::Function, constraints::Nothing, ϵ_samples::AbstractArray{<:Real}, best_yet::Nothing) =
    acq(x) = 0.
(ei::ExpectedImprovement)(fitness::Fitness, posterior::Function, constraints::AbstractVector{<:Real}, ϵ_samples::AbstractArray{<:Real}, best_yet::Nothing) =
    acq(x) = feas_prob(posterior(x)..., constraints)
(ei::ExpectedImprovement)(fitness::Fitness, posterior::Function, constraints::Nothing, ϵ_samples::AbstractArray{<:Real}, best_yet::Real) =
    acq(x) = expected_improvement(fitness, posterior(x)..., ϵ_samples, best_yet)
(ei::ExpectedImprovement)(fitness::Fitness, posterior::Function, constraints::AbstractVector{<:Real}, ϵ_samples::AbstractArray{<:Real}, best_yet::Real) =
    function acq(x)
        mean, var = posterior(x)
        ei_ = expected_improvement(fitness, mean, var, ϵ_samples, best_yet)
        fp_ = feas_prob(mean, var, constraints)
        return ei_ * fp_
    end

# Construct averaged acquisition function from multiple sampled posteriors.
function (ei::ExpectedImprovement)(fitness::Fitness, posteriors::AbstractVector{<:Function}, constraints::AbstractVector{<:Real}, ϵ_samples::AbstractMatrix{<:Real}, best_yet::Union{Nothing, <:Real})
    acqs = ei.(Ref(fitness), posteriors, Ref(constraints), eachcol(ϵ_samples), Ref(best_yet))
    x -> mapreduce(a_ -> a_(x), +, acqs) / length(acqs)
end

# Analytical EI for linear fitness function.
function expected_improvement(fitness::LinFitness, mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, ϵ_samples::AbstractArray{<:Real}, best_yet::Real)
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7352306 Eq(44)
    μf = fitness.coefs' * mean
    σf = sqrt((fitness.coefs .^ 2)' * var)
    norm_ϵ = (μf - best_yet) / σf
    return (μf - best_yet) * cdf(Distributions.Normal(), norm_ϵ) + σf * pdf(Distributions.Normal(), norm_ϵ)
end

# Approximated EI for nonlinear fitness function.
function expected_improvement(fitness::NonlinFitness, mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, ϵ_samples::AbstractMatrix{<:Real}, best_yet::Real)
    pred_samples = (mean .+ (var .* ϵ) for ϵ in eachcol(ϵ_samples))
    return sum(max.(0, fitness.(pred_samples) .- best_yet)) / size(ϵ_samples)[2]
end
function expected_improvement(fitness::NonlinFitness, mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, ϵ::AbstractVector{<:Real}, best_yet::Real)
    pred_sample = mean .+ (var .* ϵ)
    return max(0, fitness(pred_sample) - best_yet)
end

feas_prob(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, constraints::Nothing) = 1.
feas_prob(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, constraints::AbstractVector{<:Real}) = prod(cdf.(Distributions.Normal.(mean, var), constraints))

ϵ_sample_count(predict::Function, ϵ_samples::Int) = ϵ_samples
ϵ_sample_count(predict::AbstractVector{<:Function}, ϵ_samples::Int) = length(predict)

sample_ϵs(y_dim::Int, sample_count::Int) = rand(Normal(), (y_dim, sample_count))

best_so_far(problem::OptimizationProblem, posteriors::AbstractVector{<:Function}) =
    best_so_far(problem, average_posteriors(posteriors))  # TODO: better solution ?

best_so_far(problem::OptimizationProblem, posterior::Function) =
    best_so_far(problem.fitness, problem.data.X, problem.y_max, posterior)

function best_so_far(fitness::Fitness, X::AbstractMatrix{<:Real}, y_max::AbstractVector{<:Real}, posterior::Function)
    isempty(X) && return nothing
    Y_hat = mapreduce(x -> posterior(x)[1], hcat, eachcol(X))
    feasible = is_feasible.(eachcol(Y_hat), Ref(y_max))
    any(feasible) || return nothing
    maximum((fitness(Y_hat[:,i]) for i in 1:size(Y_hat)[2] if feasible[i]))
end
