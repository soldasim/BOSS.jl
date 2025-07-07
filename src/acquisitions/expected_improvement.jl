
"""
    ExpectedImprovement(; kwargs...)

The expected improvement (EI) acquisition function.

Measures the quality of a potential evaluation point `x` as the expected improvement
in best-so-far achieved `fitness` by evaluating the objective function at `y = f(x)`.

In case of constrained problems, the expected improvement is additionally weighted
by the probability of feasibility of `y`. I.e. the probability that `all(cons(y) .> 0.)`.

If the problem is constrained on `y` and no feasible point has been observed yet,
then the probability of feasibility alone is returned as the acquisition function.

Rather than using the actual evaluations `(xᵢ,yᵢ)` from the dataset,
the best-so-far achieved fitness is calculated as the maximum fitness
among the means `ŷᵢ` of the posterior predictive distribution of the model
evaluated at `xᵢ`. This is a simple way to handle evaluation noise which may not
be suitable for problems with substantial noise. In case of Bayesian Inference,
an averaged posterior of the model posterior samples is used for the prediction of `ŷᵢ`.

# Keywords
- `fitness::Fitness`: The fitness function mapping the output `y` to the real-valued score.
- `ϵ_samples::Int`: Controls how many samples are used to approximate EI.
        The `ϵ_samples` keyword is *ignored* unless `MAP` model fitter and `NonlinFitness` are used!
        In case of `BI` model fitter, the number of samples is instead set equal to the number of posterior samples.
        In case of `LinearFitness`, the expected improvement can be calculated analytically.
- `cons_safe::Bool`: If set to true, the acquisition function `acq(x)` is made 'constraint-safe'
        by checking the bounds and constraints during each evaluation.
        Set `cons_safe` to `true` if the evaluation of the model at exterior points
        may cause errors or nonsensical values.
        You may set `cons_safe` to `false` if the evaluation of the model at exterior points
        can provide useful information to the acquisition maximizer and does not cause errors.
        Defaults to `true`.
"""
@kwdef struct ExpectedImprovement{
    F<:Fitness,
} <: AcquisitionFunction
    fitness::F
    ϵ_samples::Int = 200  # only used in case of MAP and NonlinFitness
    cons_safe::Bool = true
end

get_fitness(ei::ExpectedImprovement) = ei.fitness

function construct_acquisition(ei::ExpectedImprovement, problem::BossProblem, options::BossOptions)
    post = model_posterior(problem)
    ϵ_samples = sample_ϵs(y_dim(problem), ϵ_sample_count(post, ei.ϵ_samples))
    b = best_so_far(problem, ei.fitness)
    options.info && isnothing(b) && @warn "No feasible solution in the dataset yet. Cannot calculate EI!"
    acq = construct_ei(ei.fitness, post, problem.y_max, ϵ_samples, b)
    return ei.cons_safe ? make_safe(acq, problem.domain) : acq
end

function make_safe(acq::Function, domain::Domain)
    return function acq_safe(x)
        in_bounds(x, domain.bounds) || return 0.
        in_cons(x, domain.cons) || return 0.
        # `in_discrete` ignored
        return acq(x)
    end
end

# Construct acquisition function from a single posterior.
function construct_ei(fitness::Fitness, post::ModelPosterior, constraints::Nothing, ϵ_samples::AbstractArray{<:Real}, best_yet::Nothing)
    acq(x) = 0.
end
function construct_ei(fitness::Fitness, post::ModelPosterior, constraints::AbstractVector{<:Real}, ϵ_samples::AbstractArray{<:Real}, best_yet::Nothing)
    acq(x) = feas_prob(mean_and_var(post, x)..., constraints)
end
function construct_ei(fitness::Fitness, post::ModelPosterior, constraints::Nothing, ϵ_samples::AbstractArray{<:Real}, best_yet::Real)
    acq(x) = expected_improvement(fitness, mean_and_var(post, x)..., ϵ_samples, best_yet)
end
function construct_ei(fitness::Fitness, post::ModelPosterior, constraints::AbstractVector{<:Real}, ϵ_samples::AbstractArray{<:Real}, best_yet::Real)
    function acq(x)
        μ, σ2 = mean_and_var(post, x)
        ei_ = expected_improvement(fitness, μ, σ2, ϵ_samples, best_yet)
        fp_ = feas_prob(μ, σ2, constraints)
        return ei_ * fp_
    end
end

# Construct averaged acquisition function from multiple sampled posteriors.
function construct_ei(fitness::Fitness, posteriors::AbstractVector{<:ModelPosterior}, constraints::Union{Nothing, AbstractVector{<:Real}}, ϵ_samples::AbstractMatrix{<:Real}, best_yet::Union{Nothing, Real})
    acqs = construct_ei.(Ref(fitness), posteriors, Ref(constraints), eachcol(ϵ_samples), Ref(best_yet))
    x -> mapreduce(a_ -> a_(x), +, acqs) / length(acqs)
end

# Analytical EI for linear fitness function.
function expected_improvement(fitness::LinFitness, mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, ϵ_samples::AbstractArray{<:Real}, best_yet::Real)
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7352306 Eq(44)
    μf = fitness.coefs' * mean
    σf = sqrt((fitness.coefs .^ 2)' * (var))
    diff = (μf - best_yet)
    (diff == 0. && σf == 0.) && return 0.
    norm_ϵ = diff / σf
    return diff * cdf(Distributions.Normal(), norm_ϵ) + σf * pdf(Distributions.Normal(), norm_ϵ)
end

# Approximated EI for nonlinear fitness function.
function expected_improvement(fitness::NonlinFitness, mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, ϵ_samples::AbstractMatrix{<:Real}, best_yet::Real)
    pred_samples = (mean .+ (sqrt.(var) .* ϵ) for ϵ in eachcol(ϵ_samples))
    return sum(max.(0, fitness.(pred_samples) .- best_yet)) / size(ϵ_samples)[2]
end
function expected_improvement(fitness::NonlinFitness, mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, ϵ::AbstractVector{<:Real}, best_yet::Real)
    pred_sample = mean .+ (sqrt.(var) .* ϵ)
    return max(0, fitness(pred_sample) - best_yet)
end

feas_prob(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, constraints::Nothing) = 1.
feas_prob(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, constraints::AbstractVector{<:Real}) = prod(cdf.(Distributions.Normal.(mean, sqrt.(var)), constraints))

ϵ_sample_count(post::ModelPosterior, ϵ_samples::Int) = ϵ_samples
ϵ_sample_count(post::AbstractVector{<:ModelPosterior}, ϵ_samples::Int) = length(post)

sample_ϵs(y_dim::Int, sample_count::Int) = rand(Normal(), (y_dim, sample_count))

best_so_far(problem::BossProblem, fitness::Fitness) =
    best_so_far(fitness, problem.data.X, problem.data.Y, problem.y_max)

# # noisy EI
# # Causes point clustering, needs further inspection.
# function best_so_far(fitness::Fitness, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, y_max::AbstractVector{<:Real}, posterior::Function)
#     isempty(X) && return nothing
#     Y_hat = mapreduce(x -> posterior(x)[1], hcat, eachcol(X))[:,:]
#     feasible = is_feasible.(eachcol(Y_hat), Ref(y_max))
#     any(feasible) || return nothing
#     maximum((fitness(Y_hat[:,i]) for i in 1:size(Y_hat)[2] if feasible[i]))
# end

# classic EI
function best_so_far(fitness::Fitness, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, y_max::AbstractVector{<:Real})
    isempty(Y) && return nothing
    feasible = is_feasible.(eachcol(Y), Ref(y_max))
    any(feasible) || return nothing
    maximum((fitness(Y[:,i]) for i in 1:size(Y)[2] if feasible[i]))
end
