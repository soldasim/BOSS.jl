using Distributions
using Optim
using Turing
using Turing: Variational
using FLoops
using Zygote
using Evolutionary

# - - - - - - OPTIMIZATION - - - - - -

# Find ̂x that maximizes f(x) using parallel multistart LBFGS.
function optim_params(f, starts; kwargs...)
    return optim_params(f, starts, nothing; kwargs...)
end
function optim_params(f, starts, constraints; options=Optim.Options(; x_abstol=1e-2, iterations=200), info=true, debug=false)
    multistart = size(starts)[2]

    args = Vector{Vector{Float64}}(undef, multistart)
    vals = Vector{Float64}(undef, multistart)
    convergence_errors = 0
    @floop for i in 1:multistart
        try
            a, v = optim_(f, starts[:,i], constraints; options, info)
            args[i] = a
            vals[i] = v
        catch e
            debug && throw(e)
            @reduce convergence_errors += 1
            args[i] = Float64[]
            vals[i] = -Inf
        end
    end
    info && (convergence_errors > 0) && print("      $(convergence_errors)/$(multistart) optimization runs failed to converge!\n")
    
    opt_i = argmax(vals)
    return args[opt_i], vals[opt_i]
end

# TODO try changing optimization algorithms
function optim_(f, start, constraints::Nothing; options=Optim.Options(; x_abstol=1e-2, iterations=200), info=false)
    opt_res = Optim.optimize(p -> -f(p), start, NelderMead(), options)
    info && (Optim.iterations(opt_res) == options.iterations) && println("Warning: Maximum iterations reached while optimizing the acquisition function!")
    return Optim.minimizer(opt_res), -Optim.minimum(opt_res)
end
function optim_(f, start, constraints::Tuple; options=Optim.Options(; x_abstol=1e-2, iterations=200), info=false)
    lb, ub = constraints
    opt_res = Optim.optimize(p -> -f(p), lb, ub, start, Fminbox(LBFGS()), options)
    info && (Optim.iterations(opt_res) == options.iterations) && println("Warning: Maximum iterations reached while optimizing the acquisition function!")
    return Optim.minimizer(opt_res), -Optim.minimum(opt_res)
end
function optim_(f, start, constraints::TwiceDifferentiableConstraints; options=Optim.Options(; x_abstol=1e-2, iterations=200), info=false)
    opt_res = Optim.optimize(p -> -f(p), constraints, start, IPNewton(), options)
    info && (Optim.iterations(opt_res) == options.iterations) && println("Warning: Maximum iterations reached while optimizing the acquisition function!")
    return Optim.minimizer(opt_res), -Optim.minimum(opt_res)
end
function optim_(f, start, constraints::Evolutionary.AbstractConstraints; kwargs...)
    throw(ArgumentError("`$(typeof(constraints))` are not supported with Optim. Use `Optim.TwiceDifferentiableConstraints` instead."))
end

# - - - - - - CMA-ES - - - - - -

# maximize f(x)
optim_cmaes(f, start; kwargs...) = optim_cmaes(f, Evolutionary.NoConstraints(), start; kwargs...)
optim_cmaes(f, bounds::Tuple, start; kwargs...) = optim_cmaes(f, Evolutionary.BoxConstraints(bounds...), start; kwargs...)
function optim_cmaes(f, constraints::Evolutionary.AbstractConstraints, start; options=Evolutionary.Options(; abstol=1e-5, iterations=800), info=false)
    res = Evolutionary.optimize(x->-f(x), constraints, start, CMAES(), options)
    info && (Evolutionary.iterations(res) == options.iterations) && println("Warning: Maximum iterations reached while optimizing the acquisition function!")
    return res.minimizer, -res.minimum
end
function optim_cmaes(f, cons::Optim.TwiceDifferentiableConstraints, start; kwargs...)
    throw(ArgumentError("`Optim.TwiceDifferentiableConstraints` are not supported with CMAES. Use `Evolutionary.WorstFitnessConstraints` or any other constraints from the `Evolutionary` package instead."))
end

function optim_cmaes_multistart(f, constraints, starts; options=Evolutionary.Options(; abstol=1e-5, iterations=800), info=false)
    multistart = size(starts)[2]

    args = Vector{Vector{Float64}}(undef, multistart)
    vals = Vector{Float64}(undef, multistart)
    @floop for i in 1:multistart
        a, v = optim_cmaes(f, constraints, starts[:,i]; options, info)
        args[i] = a
        vals[i] = v
    end
    
    opt_i = argmax(vals)
    return args[opt_i], vals[opt_i]
end

# - - - - - - NUTS SAMPLING - - - - - -

"""
Stores hyperparameters of the MC sampler.

Amount of drawn samples:    'chain_count * (warmup + leap_size * sample_count)'
Amount of used samples:     'chain_count * sample_count'

# Fields
  - warmup: The amount of initial unused 'warmup' samples in each chain.
  - sample_count: The amount of samples used from each chain.
  - chain_count: The amount of independent chains sampled.
  - leap_size: The "distance" between two following used samples in a chain. (To avoid correlated samples.)

In each chain;
    Firstly, the first 'warmup' samples are discarded.
    Then additional 'leap_size * sample_count' samples are drawn
    and each 'leap_size'-th of these samples is kept.
Finally, kept samples from all chains are joined and returned.
"""
struct MCSettings
    warmup::Int
    samples_in_chain::Int
    chain_count::Int
    leap_size::Int
end

sample_count(mc::MCSettings) = mc.chain_count * mc.samples_in_chain

# Sample parameters of the given probabilistic model (defined with Turing.jl) using parallel NUTS MC sampling.
# Other AD backends than Zygote cause issues: https://discourse.julialang.org/t/gaussian-process-regression-with-turing-gets-stuck/86892
function sample_params_turing(model, param_symbols, mc::MCSettings; adbackend=:zygote)
    Turing.setadbackend(adbackend)

    # # NUTS
    # samples_in_chain = mc.leap_size * mc.samples_in_chain
    # chains = Turing.sample(model, NUTS(mc.warmup, 0.65), MCMCThreads(), samples_in_chain, mc.chain_count; progress=false)
    # samples = [vec(chains[s][mc.leap_size:mc.leap_size:end,:]) for s in param_symbols]

    # PG
    # (The built-in parallelization in Turing occasionally causes 'access to undefined reference' errors.)
    samples_in_chain = mc.warmup + (mc.leap_size * mc.samples_in_chain)
    chains = Vector{Turing.Chains}(undef, mc.chain_count)
    @floop for i in 1:mc.chain_count
        chains[i] = Turing.sample(model, PG(20), samples_in_chain; progress=false)
    end
    samples = [reduce(vcat, [ch[s][(mc.warmup+mc.leap_size):mc.leap_size:end,:] for ch in chains]) for s in param_symbols]

    return samples
end

# TODO unused
function sample_params_vi(model, samples; alg=ADVI{Turing.AdvancedVI.ForwardDiffAD{0}}(10, 1000))
    posterior = vi(model, alg)
    rand(posterior, samples) |> eachrow |> collect
end

# - - - - - - DATA GENERATION - - - - - -

# Sample from uniform distribution.
function uniform_sample(a, b, sample_size)
    distr = Product(Distributions.Uniform.(a, b))
    X = rand(distr, sample_size)
    return vec.(collect.(eachslice(X; dims=length(size(X)))))
end

# Sample from log-uniform distribution.
function log_sample(a, b, sample_size)
    X = [exp.(x) for x in uniform_sample(log.(a), log.(b), sample_size)]
    return X
end

# Return points distributed evenly over a given logarithmic range.
function log_range(a, b, len)
    a = log10.(a)
    b = log10.(b)
    range = collect(LinRange(a, b, len))
    range = [10 .^ range[i] for i in 1:len]
    return range
end

# - - - - - - MODEL ERROR - - - - - -

# Calculate RMS error with given y values and models predictions for them.
function rms_error(preds, ys; N=nothing)
    isnothing(N) && (N = length(preds))
    return sqrt((1 / N) * sum((preds .- ys).^2))
end
# Calculate RMS error with given test data.
function rms_error(X, Y, model)
    dims = size(Y)[2]
    preds = reduce(hcat, model.(eachrow(X)))'
    return [rms_error(preds[:,i], Y[:,i]) for i in 1:dims]
end
# Calculate RMS error using uniformly sampled test data.
function rms_error(obj_func, model, a, b, sample_count)
    X = reduce(hcat, uniform_sample(a, b, sample_count))'
    Y = reduce(hcat, obj_func.(X))'
    return rms_error(X, Y, model)
end

function partial_sums(array)
    isempty(array) && return empty(array)
    
    s = zero(first(array))
    sums = [(s += val) for val in array]
    return sums
end

# - - - - - - OTHER - - - - - -

get_bounds(bounds::Tuple) = bounds
get_bounds(constraints::TwiceDifferentiableConstraints) = get_bounds_std_format_(constraints)
get_bounds(constraints::Evolutionary.AbstractConstraints) = get_bounds_std_format_(constraints)
get_bounds(constraints::MixedTypePenaltyConstraints) = get_bounds(constraints.penalty)

function get_bounds_std_format_(constraints)
    domain_lb = constraints.bounds.bx[1:2:end]
    domain_ub = constraints.bounds.bx[2:2:end]
    return domain_lb, domain_ub
end

function in_domain(x::AbstractVector, domain::Tuple)
    lb, ub = domain
    any(x .< lb) && return false
    any(x .> ub) && return false
    return true
end
in_domain(x::AbstractVector, domain::TwiceDifferentiableConstraints) = Optim.isinterior(domain, x)
in_domain(x::AbstractVector, domain::Evolutionary.AbstractConstraints) = Evolutionary.isfeasible(domain, x)

Evolutionary.isfeasible(c::MixedTypePenaltyConstraints, x) = Evolutionary.isfeasible(c.penalty, x)
