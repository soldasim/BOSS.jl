using Optim
using LatinHypercubeSampling

abstract type Fitness end

"""
Used to define a linear fitness function for the BOSS algorithm.

# Example
A fitness function 'f(y) = y[1] + a * y[2] + b * y[3]' can be defined as:
```julia-repl
julia> LinFitness([1., a, b])
```
"""
struct LinFitness <: Fitness
    coefs::Vector{Float64}
end
function (f::LinFitness)(y)
    return f.coefs' * y
end

"""
Used to define a fitness function for the BOSS algorithm.
If possible, the 'LinFitness' option should be used instead for a better performance.

# Example
```julia-repl
julia> NonlinFitness(y -> cos(y[1]) + sin(y[2]))
```
"""
struct NonlinFitness <: Fitness
    fitness::Function
end
function (f::NonlinFitness)(y)
    return f.fitness(y)
end

function construct_acq(fitness::Fitness, model, constraints::Nothing, ϵ_samples::AbstractArray{<:Real}, best_yet::Nothing)
    print("WARNING: No feasible solution in the dataset yet! Cannot calculate EI.\n")
    acq(x) = 0.
end
function construct_acq(fitness::Fitness, model, constraints::Nothing, ϵ_samples::AbstractArray{<:Real}, best_yet)
    acq(x) = EI(x, fitness, model, ϵ_samples; best_yet)
end
function construct_acq(fitness::Fitness, model, constraints::AbstractVector{<:Real}, ϵ_samples::AbstractArray{<:Real}, best_yet::Nothing)    
    print("WARNING: No feasible solution in the dataset yet! Cannot calculate EI.\n")
    acq(x) = feas_prob(x, model, constraints)
end
function construct_acq(fitness::Fitness, model, constraints::AbstractVector{<:Real}, ϵ_samples::AbstractArray{<:Real}, best_yet)    
    function acq(x)
        mean, var = model(x)
        ei = EI(mean, var, fitness, ϵ_samples; best_yet)
        fp = feas_prob(mean, var, constraints)
        ei * fp
    end
end

function construct_acq_from_samples(fitness::Fitness, models::AbstractArray, constraints, ϵ_samples::AbstractMatrix{<:Real}, best_yet)
    acqs = construct_acq.(Ref(fitness), models, Ref(constraints), eachcol(ϵ_samples), best_yet)
    acq(x) = mapreduce(a -> a(x), +, acqs) / length(acqs)
end

feas_prob(x::AbstractVector{<:Real}, model, constraints) = feas_prob(model(x)..., constraints)
feas_prob(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, constraints::Nothing) = 1.
feas_prob(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, constraints::AbstractVector{<:Real}) = prod(cdf.(Distributions.Normal.(mean, var), constraints))

EI(x::AbstractVector{<:Real}, fitness::Fitness, model, ϵ_samples::AbstractArray{<:Real}; best_yet) = EI(model(x)..., fitness, ϵ_samples; best_yet)

EI(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, fitness::LinFitness, ϵ_samples::AbstractArray{<:Real}; best_yet) = EI(mean, var, fitness; best_yet)
function EI(mean, var, fitness::LinFitness; best_yet)
    μf = fitness.coefs' * mean
    σf = sqrt((fitness.coefs .^ 2)' * var)
    
    norm_ϵ = (μf - best_yet) / σf
    return (μf - best_yet) * cdf(Distributions.Normal(), norm_ϵ) + σf * pdf(Distributions.Normal(), norm_ϵ)
end

function EI(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, fitness::NonlinFitness, ϵ_samples::AbstractMatrix{<:Real}; best_yet::AbstractVector{<:Real})
    pred_samples = [mean .+ (var .* ϵ) for ϵ in eachcol(ϵ_samples)]
    return sum(max.(0, fitness.(pred_samples) .- best_yet)) / size(ϵ_samples)[2]
end
function EI(mean::AbstractVector{<:Real}, var::AbstractVector{<:Real}, fitness::NonlinFitness, ϵ::AbstractVector{<:Real}; best_yet::Real)
    pred_sample = mean .+ (var .* ϵ)
    return max(0, fitness(pred_sample) - best_yet)
end

function opt_acq_Optim(acq, domain; x_dim::Int, multistart=1, discrete_dims=nothing, options, parallel, info=true, debug=false)
    # starts = generate_starts_random_(domain, multistart)
    starts = generate_starts_LHC_(domain, multistart; x_dim)
    arg, val = optim_params(acq, starts, domain; options, parallel, info, debug)
   
    isnothing(discrete_dims) || (arg = discrete_round(discrete_dims)(arg))
    return arg, val
end
function opt_acq_CMAES(acq, domain; x_dim::Int, multistart=1, discrete_dims=nothing, options, parallel, info=true, debug=false)
    # starts = generate_starts_random_(domain, multistart)
    starts = generate_starts_LHC_(domain, multistart; x_dim)

    arg, val = optim_cmaes_multistart(acq, domain, starts; options, parallel, info)

    isnothing(discrete_dims) || (arg = discrete_round(discrete_dims)(arg))
    return arg, val
end

function generate_starts_LHC_(domain::Tuple, count::Int; x_dim::Int)
    @assert count > 1  # `randomLHC(count, dim)` returns NaNs if `count == 1`
    lb, ub = domain
    starts = scaleLHC(randomLHC(count, x_dim), [(lb[i], ub[i]) for i in 1:x_dim])'
    return starts
end
# TODO: Implement LHC generation for more complex domains.
generate_starts_LHC_(domain, count::Int; x_dim::Int) = generate_starts_LHC_(get_bounds(domain), count; x_dim)

function generate_starts_random_(domain, count::Int)
    return reduce(vcat, transpose.([generate_starting_point_(domain) for _ in 1:count]))
end

function generate_start_(domain::Tuple)
    lb, ub = domain
    dim = length(lb)
    start = rand(dim) .* (ub .- lb) .+ lb
    return start
end
function generate_start_(domain)
    bounds = get_bounds(domain)
    start = nothing
    while isnothing(start) || (!in_domain(start, domain))
        start = generate_start_(bounds)
    end
    return start
end
