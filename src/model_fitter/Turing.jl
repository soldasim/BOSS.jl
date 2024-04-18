
"""
    TuringBI(; kwargs...)

Samples the model parameters and hyperparameters using the Turing.jl package.

# Keywords
- `sampler::Any`: The sampling algorithm used to draw the samples.
- `n_adapts::Int`: The amount of initial unused 'warmup' samples in each chain.
- `samples_in_chain::Int`: The amount of samples used from each chain.
- `chain_count::Int`: The amount of independent chains sampled.
- `leap_size`: Every `leap_size`-th sample is used from each chain. (To avoid correlated samples.)
- `parallel`: If `parallel=true` then the chains are sampled in parallel.

# Sampling Process

In each sampled chain;
  - The first `n_adapts` samples are discarded.
  - From the following `leap_size * samples_in_chain` samples each `leap_size`-th is kept.
Then the samples from all chains are concatenated and returned.

Total drawn samples:    'chain_count * (warmup + leap_size * samples_in_chain)'
Total returned samples: 'chain_count * samples_in_chain'
"""
struct TuringBI{S} <: ModelFitter{BI}
    sampler::S
    n_adapts::Int
    samples_in_chain::Int
    chain_count::Int
    leap_size::Int
    parallel::Bool
end
TuringBI(;
    sampler=PG(20),
    warmup=400,
    samples_in_chain=10,
    chain_count=8,
    leap_size=5,
    parallel=true,
) = TuringBI(sampler, warmup, samples_in_chain, chain_count, leap_size, parallel)

function estimate_parameters(turing::TuringBI, problem::BossProblem, options::BossOptions)
    return sample_params(turing, problem.model, problem.noise_var_priors, problem.data.X, problem.data.Y)
end

Turing.@model function turing_model(
    model::Parametric,
    noise_var_priors::AbstractVector{<:UnivariateDistribution},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    noise_vars ~ product_distribution(noise_var_priors)
    θ ~ product_distribution(model.param_priors)

    means = model.(eachcol(X), Ref(θ))
    
    Y ~ product_distribution(Distributions.MvNormal.(means, Ref(noise_vars)))
end
Turing.@model function turing_model(
    model::GaussianProcess,
    noise_var_priors::AbstractVector{<:UnivariateDistribution},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    y_dim = size(Y)[1]

    noise_vars ~ product_distribution(noise_var_priors)
    length_scales ~ product_distribution(model.length_scale_priors)
    
    gps = [finite_gp(model.mean, model.kernel, X, length_scales[:,i], noise_vars[i]) for i in 1:y_dim]

    Yt = transpose(Y)
    Yt ~ product_distribution(gps)
end
Turing.@model function turing_model(
    model::Semiparametric,
    noise_var_priors::AbstractVector{<:UnivariateDistribution},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    y_dim = size(Y)[1]

    noise_vars ~ product_distribution(noise_var_priors)
    θ ~ product_distribution(model.parametric.param_priors)
    length_scales ~ product_distribution(model.nonparametric.length_scale_priors)

    mean = model.parametric(θ)
    gps = [finite_gp(x->mean(x)[i], model.nonparametric.kernel, X, length_scales[:,i], noise_vars[i]) for i in 1:y_dim]
    
    Yt = transpose(Y)
    Yt ~ product_distribution(gps)
end

function sample_params(
    turing::TuringBI,
    model::Parametric,
    noise_var_priors::AbstractVector{<:UnivariateDistribution},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    x_dim, y_dim = size(X)[1], size(Y)[1]
    θ_len, λ_len = param_counts(model)
    
    tm = turing_model(model, noise_var_priors, X, Y)
    param_symbols = vcat(
        [Symbol("noise_vars[$i]") for i in 1:y_dim],
        [Symbol("θ[$i]") for i in 1:θ_len],
    )

    samples = sample_params_turing(turing, tm, param_symbols)
    noise_vars = reduce(vcat, transpose.(samples[1:y_dim]))
    θ = reduce(vcat, transpose.(samples[y_dim+1:end]))
    return θ, nothing, noise_vars
end
function sample_params(
    turing::TuringBI,
    model::GaussianProcess,
    noise_var_priors::AbstractVector{<:UnivariateDistribution},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    x_dim, y_dim = size(X)[1], size(Y)[1]
    
    tm = turing_model(model, noise_var_priors, X, Y)
    param_symbols = vcat(
        [Symbol("noise_vars[$i]") for i in 1:y_dim],
        [[Symbol("length_scales[$j,$i]") for j in 1:x_dim] for i in 1:y_dim] |> x->reduce(vcat,x),
    )

    samples = sample_params_turing(turing, tm, param_symbols)
    noise_vars = reduce(vcat, transpose.(samples[1:y_dim]))
    length_scales = reshape.(eachcol(reduce(vcat, transpose.(samples[y_dim+1:end]))), Ref(x_dim), Ref(y_dim))
    return nothing, length_scales, noise_vars
end
function sample_params(
    turing::TuringBI,
    model::Semiparametric,
    noise_var_priors::AbstractVector{<:UnivariateDistribution},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)    
    x_dim, y_dim = size(X)[1], size(Y)[1]
    θ_len, λ_len = param_counts(model)
    
    tm = turing_model(model, noise_var_priors, X, Y)
    param_symbols = vcat(
        [Symbol("noise_vars[$i]") for i in 1:y_dim],
        [Symbol("θ[$i]") for i in 1:θ_len],
        [[Symbol("length_scales[$j,$i]") for j in 1:x_dim] for i in 1:y_dim] |> x->reduce(vcat,x),
    )

    samples = sample_params_turing(turing, tm, param_symbols)
    noise_vars = reduce(vcat, transpose.(samples[1:y_dim]))
    θ = reduce(vcat, transpose.(samples[y_dim+1:y_dim+θ_len]))
    length_scales = reshape.(eachcol(reduce(vcat, transpose.(samples[y_dim+θ_len+1:end]))), Ref(x_dim), Ref(y_dim))
    return θ, length_scales, noise_vars
end

function sample_params_turing(turing::TuringBI, turing_model, param_symbols)
    samples_in_chain = turing.n_adapts + (turing.leap_size * turing.samples_in_chain)
    if turing.parallel
        chains = Turing.sample(turing_model, turing.sampler, MCMCThreads(), samples_in_chain, turing.chain_count; progress=false)
    else
        chains = mapreduce(_ -> Turing.sample(turing_model, turing.sampler, samples_in_chain; progress=false), chainscat, 1:turing.chain_count)
    end

    samples = [reduce(vcat, eachrow(chains[s][(turing.n_adapts+turing.leap_size):turing.leap_size:end,:])) for s in param_symbols]
end
