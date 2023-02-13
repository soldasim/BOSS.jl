
sample_ϵs(y_dim::Int, sample_count::Int) = rand(Distributions.Normal(), (y_dim, sample_count))

acquisition(fitness::Fitness, posterior::Base.Callable, constraints::Nothing, ϵ_samples::AbstractArray{<:Real}, best_yet::Nothing) =
    acq(x) = 0.

acquisition(fitness::Fitness, posterior::Base.Callable, constraints::AbstractVector{<:Real}, ϵ_samples::AbstractArray{<:Real}, best_yet::Nothing) =
    acq(x) = feas_prob(x, posterior, constraints)

acquisition(fitness::Fitness, posterior::Base.Callable, constraints::Nothing, ϵ_samples::AbstractArray{<:Real}, best_yet::Real) =
    acq(x) = EI(x, fitness, posterior, ϵ_samples; best_yet)

acquisition(fitness::Fitness, posterior::Base.Callable, constraints::AbstractVector{<:Real}, ϵ_samples::AbstractArray{<:Real}, best_yet::Real) =
    function acq(x)
        mean, var = posterior(x)
        ei = EI(mean, var, fitness, ϵ_samples; best_yet)
        fp = feas_prob(mean, var, constraints)
        ei * fp
    end

function acquisition(fitness::Fitness, posteriors::AbstractArray{<:Base.Callable}, constraints::AbstractArray{<:Real}, ϵ_samples::AbstractMatrix{<:Real}, best_yet::Union{Nothing, <:Real})
    acqs = acquisition.(Ref(fitness), posteriors, Ref(constraints), eachcol(ϵ_samples), Ref(best_yet))
    acq(x) = mapreduce(a -> a(x), +, acqs) / length(acqs)
end

feas_prob(x::AbstractVector{<:Real}, posterior, constraints) = feas_prob(posterior(x)..., constraints)
feas_prob(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, constraints::Nothing) = 1.
feas_prob(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, constraints::AbstractVector{<:Real}) = prod(cdf.(Distributions.Normal.(mean, var), constraints))

EI(x::AbstractVector{<:Real}, fitness::Fitness, posterior, ϵ_samples::AbstractArray{<:Real}; best_yet) = EI(posterior(x)..., fitness, ϵ_samples; best_yet)

EI(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, fitness::LinFitness, ϵ_samples::AbstractArray{<:Real}; best_yet) = EI(mean, var, fitness; best_yet)
# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7352306 Eq(44)
function EI(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, fitness::LinFitness; best_yet::Real)
    μf = fitness.coefs' * mean
    σf = sqrt((fitness.coefs .^ 2)' * var)
    
    norm_ϵ = (μf - best_yet) / σf
    return (μf - best_yet) * cdf(Distributions.Normal(), norm_ϵ) + σf * pdf(Distributions.Normal(), norm_ϵ)
end

function EI(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, fitness::NonlinFitness, ϵ_samples::AbstractMatrix{<:Real}; best_yet::Real)
    pred_samples = [mean .+ (var .* ϵ) for ϵ in eachcol(ϵ_samples)]
    return sum(max.(0, fitness.(pred_samples) .- best_yet)) / size(ϵ_samples)[2]
end
function EI(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, fitness::NonlinFitness, ϵ::AbstractVector{<:Real}; best_yet::Real)
    pred_sample = mean .+ (var .* ϵ)
    return max(0, fitness(pred_sample) - best_yet)
end
