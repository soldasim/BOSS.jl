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

function construct_acq(acq, feas_probs; feasibility, best_yet)
    if feasibility
        if isnothing(best_yet)
            return x -> prod(feas_probs(x))
        else
            return x -> prod(feas_probs(x)) * acq(x)
        end

    else
        if isnothing(best_yet)
            # TODO better solution (May be unnecessary as this case is rare and can only happen in the zero-th iteration.)
            print("WARNING: No feasible solution in the dataset yet! Cannot calculate EI.\n")
            return x -> 0.
        else
            return acq
        end
    end
end

function feasibility_probabilities(feasibility_model)
    function p(x)
        μ, σ = feasibility_model(x)
        N = length(μ)
        [(1. - cdf(Distributions.Normal(μ[i], σ[i]), 0.)) for i in 1:N]
    end
end

function EI(x, fitness::LinFitness, model; best_yet)
    mean, var = model(x)

    μf = fitness.coefs' * mean
    σf = sqrt((fitness.coefs .^ 2)' * var)
    
    norm_ϵ = (μf - best_yet) / σf
    return (μf - best_yet) * cdf(Distributions.Normal(), norm_ϵ) + σf * pdf(Distributions.Normal(), norm_ϵ)
end
EI(x, fitness::LinFitness, model, ϵ_samples; best_yet) = EI(x, fitness, model; best_yet)

function EI(x, fitness::NonlinFitness, model, ϵ_samples::AbstractMatrix{<:Real}; best_yet)
    mean, var = model(x)
    pred_samples = [mean .+ (var .* ϵ) for ϵ in eachcol(ϵ_samples)]
    return sum(max.(0, fitness.(pred_samples) .- best_yet)) / size(ϵ_samples)[2]
end
function EI(x, fitness::NonlinFitness, model, ϵ_sample::AbstractArray{<:Real}; best_yet)
    mean, var = model(x)
    pred_sample = mean .+ (var .* ϵ_sample)
    return max(0, fitness(pred_sample) - best_yet)
end

EI_sampled(x, fitness, model_samples, ϵ_samples; best_yet) = mapreduce(mϵ -> EI(x, fitness, mϵ...; best_yet), +, zip(model_samples, ϵ_samples))

function opt_acq_Optim(acq, domain; x_dim, multistart=1, discrete_dims=nothing, info=true, debug=false)
    # starts = generate_starts_random_(domain, multistart)
    starts = generate_starts_LHC_(domain, multistart; x_dim)
    arg, val = optim_params(acq, starts, domain; info, debug)
   
    isnothing(discrete_dims) || (arg = discrete_round(discrete_dims)(arg))
    return arg, val
end
function opt_acq_CMAES(acq, domain; x_dim, multistart=1, discrete_dims=nothing, info=true, debug=false)
    # starts = generate_starts_random_(domain, multistart)
    starts = generate_starts_LHC_(domain, multistart; x_dim)

    arg, val = optim_cmaes_multistart(acq, domain, starts)

    isnothing(discrete_dims) || (arg = discrete_round(discrete_dims)(arg))
    return arg, val
end

function generate_starts_LHC_(domain::Tuple, count; x_dim)
    lb, ub = domain
    starts = scaleLHC(randomLHC(count, x_dim), [(lb[i], ub[i]) for i in 1:x_dim])'
    return starts
end
function generate_starts_LHC_(domain, count; x_dim)
    bounds = get_bounds(domain)
    starts = generate_starts_LHC_(bounds, count; x_dim)
    
    # replace exterior vertices with random points
    interior = in_domain.(eachcol(starts), Ref(domain))
    for i in 1:count
        interior[i] && continue
        starts[:,i] = generate_start_(domain)
    end

    return starts
end

function generate_starts_random_(domain, count)
    return reduce(hcat, [generate_starting_point_(domain) for _ in 1:count])'
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
