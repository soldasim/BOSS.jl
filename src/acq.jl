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
        μ, σ = feasibility_model[1](x), feasibility_model[2](x)
        N = length(μ)
        distrs = [Distributions.Normal(μ[i], σ[i]) for i in 1:N]
        return [(1. - cdf(d, 0.)) for d in distrs]
    end
    return p
end

function EI(x, fitness::LinFitness, model, ϵ_samples; best_yet, sample_count)
    μy, Σy = model[1](x), model[2](x)

    μf = fitness.coefs' * μy
    σf = sqrt((fitness.coefs .^ 2)' * (Σy .^ 2))
    
    norm_ϵ = (μf - best_yet) / σf
    return (μf - best_yet) * cdf(Distributions.Normal(), norm_ϵ) + σf * pdf(Distributions.Normal(), norm_ϵ)
end

function EI(x, fitness::NonlinFitness, model, ϵ_samples; best_yet, sample_count)
    μy, Σy = model[1](x), model[2](x)
    pred_samples = [μy .+ (Σy .* ϵ_samples[i,:]) for i in 1:sample_count]
    return sum(max.(0, fitness.(pred_samples) .- best_yet)) / sample_count
end

# 'domain' is a Tuple of lb and ub or TwiceDifferentiableConstraints
function opt_acq(acq, domain; x_dim, multistart=1, discrete_dims=nothing, info=true, debug=false)
    # starts = generate_starts_random_(domain, multistart)
    starts = generate_starts_LHC_(domain, multistart; x_dim)
    arg, val = optim_params(acq, starts, domain; info, debug)
    
    isnothing(discrete_dims) || (arg = discrete_round(discrete_dims)(arg))
    return arg, val
end

function generate_starts_LHC_(domain::Tuple, count; x_dim)
    lb, ub = domain
    starts = scaleLHC(randomLHC(count, x_dim), [(lb[i], ub[i]) for i in 1:x_dim])
    return starts
end
function generate_starts_LHC_(domain::TwiceDifferentiableConstraints, count; x_dim)
    bounds = get_bounds(domain)
    starts = generate_starts_LHC_(bounds, count; x_dim)
    
    # replace exterior vertices with random points
    interior = Optim.isinterior.(Ref(domain), collect(eachrow(starts)))
    for i in 1:count
        interior[i] && continue
        starts[i,:] = generate_start_(domain)
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
function generate_start_(domain::TwiceDifferentiableConstraints)
    bounds = get_bounds(domain)
    start = nothing
    while isnothing(start) || (!Optim.isinterior(domain, start))
        start = generate_start_(bounds)
    end
    return start
end

function get_bounds(constraints::TwiceDifferentiableConstraints)
    domain_lb = constraints.bounds.bx[1:2:end]
    domain_ub = constraints.bounds.bx[2:2:end]
    return domain_lb, domain_ub
end
function get_bounds(domain::Tuple)
    return domain
end
