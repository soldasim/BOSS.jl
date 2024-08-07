
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
    return sample_params(turing, problem.model, problem.noise_std_priors, problem.data.X, problem.data.Y)
end

Turing.@model function turing_model(
    model::Parametric,
    noise_std_priors::AbstractVector{<:UnivariateDistribution},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    θ ~ product_distribution(model.param_priors)
    noise_std ~ product_distribution(noise_std_priors)

    means = model.(eachcol(X), Ref(θ))
    
    Y ~ product_distribution(Distributions.MvNormal.(means, Ref(noise_std)))
end
Turing.@model function turing_model(
    model::GaussianProcess,
    noise_std_priors::AbstractVector{<:UnivariateDistribution},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    y_dim = size(Y)[1]

    length_scales ~ product_distribution(model.length_scale_priors)
    amplitudes ~ product_distribution(model.amp_priors)
    noise_std ~ product_distribution(noise_std_priors)
    
    gps = [finite_gp(model.mean, model.kernel, X, length_scales[:,i], amplitudes[i], noise_std[i]) for i in 1:y_dim]

    Yt = transpose(Y)
    Yt ~ product_distribution(gps)
end
Turing.@model function turing_model(
    model::Semiparametric,
    noise_std_priors::AbstractVector{<:UnivariateDistribution},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    y_dim = size(Y)[1]

    θ ~ product_distribution(model.parametric.param_priors)
    length_scales ~ product_distribution(model.nonparametric.length_scale_priors)
    amplitudes ~ product_distribution(model.nonparametric.amp_priors)
    noise_std ~ product_distribution(noise_std_priors)

    mean = model.parametric(θ)
    gps = [finite_gp(x->mean(x)[i], model.nonparametric.kernel, X, length_scales[:,i], amplitudes[i], noise_std[i]) for i in 1:y_dim]
    
    Yt = transpose(Y)
    Yt ~ product_distribution(gps)
end

function sample_params(
    turing::TuringBI,
    model::Parametric,
    noise_std_priors::AbstractVector{<:UnivariateDistribution},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    x_dim, y_dim = size(X)[1], size(Y)[1]
    θ_len, λ_len, α_len = param_counts(model)

    tm = turing_model(model, noise_std_priors, X, Y)
    chains = sample_params_turing(turing, tm)

    θs = get_samples(chains, "θ", (θ_len,))
    noise_std = get_samples(chains, "noise_std", (y_dim,))

    return θs, nothing, nothing, noise_std
end
function sample_params(
    turing::TuringBI,
    model::GaussianProcess,
    noise_std_priors::AbstractVector{<:UnivariateDistribution},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    x_dim, y_dim = size(X)[1], size(Y)[1]
    
    tm = turing_model(model, noise_std_priors, X, Y)
    chains = sample_params_turing(turing, tm)
    
    length_scales = get_samples(chains, "length_scales", (x_dim, y_dim))
    amplitudes = get_samples(chains, "amplitudes", (y_dim,))
    noise_std = get_samples(chains, "noise_std", (y_dim,))

    return nothing, length_scales, amplitudes, noise_std
end
function sample_params(
    turing::TuringBI,
    model::Semiparametric,
    noise_std_priors::AbstractVector{<:UnivariateDistribution},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)    
    x_dim, y_dim = size(X)[1], size(Y)[1]
    θ_len, λ_len, α_len = param_counts(model)
    
    tm = turing_model(model, noise_std_priors, X, Y)
    chains = sample_params_turing(turing, tm)

    θs = get_samples(chains, "θ", (θ_len,))
    length_scales = get_samples(chains, "length_scales", (x_dim, y_dim))
    amplitudes = get_samples(chains, "amplitudes", (y_dim,))
    noise_std = get_samples(chains, "noise_std", (y_dim,))

    return θs, length_scales, amplitudes, noise_std
end

function sample_params_turing(turing::TuringBI, turing_model)
    samples_in_chain = turing.n_adapts + (turing.leap_size * turing.samples_in_chain)
    if turing.parallel
        chains = Turing.sample(turing_model, turing.sampler, MCMCThreads(), samples_in_chain, turing.chain_count; progress=false)
    else
        chains = mapreduce(_ -> Turing.sample(turing_model, turing.sampler, samples_in_chain; progress=false), chainscat, 1:turing.chain_count)
    end
    chains = chains[turing.n_adapts+turing.leap_size:turing.leap_size:end]
    return chains
end

function get_samples(chains, param_name, param_shape)
    # retrieve matrix (samples × param_count × chains)
    samples = group(chains, param_name).value
    # concatenate chains & transpose into matrix (param_count × samples)
    samples = reduce(vcat, (samples[:,:,i] for i in 1:size(samples)[3])) |> transpose
    # return vector of reshaped samples
    return [reshape(s, param_shape) for s in eachcol(samples)]
end
