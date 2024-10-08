
"""
    TuringBI(; kwargs...)

Samples the model parameters and hyperparameters using the Turing.jl package.

# Keywords
- `sampler::Any`: The sampling algorithm used to draw the samples.
- `warmup::Int`: The amount of initial unused 'warmup' samples in each chain.
- `samples_in_chain::Int`: The amount of samples used from each chain.
- `chain_count::Int`: The amount of independent chains sampled.
- `leap_size`: Every `leap_size`-th sample is used from each chain. (To avoid correlated samples.)
- `parallel`: If `parallel=true` then the chains are sampled in parallel.

# Sampling Process

In each sampled chain;
  - The first `warmup` samples are discarded.
  - From the following `leap_size * samples_in_chain` samples each `leap_size`-th is kept.
Then the samples from all chains are concatenated and returned.

Total drawn samples:    'chain_count * (warmup + leap_size * samples_in_chain)'
Total returned samples: 'chain_count * samples_in_chain'
"""
struct TuringBI{S} <: ModelFitter{BI}
    sampler::S
    warmup::Int
    samples_in_chain::Int
    chain_count::Int
    leap_size::Int
    parallel::Bool
end
TuringBI(;
    sampler = PG(20),
    warmup = 400,
    samples_in_chain = 10,
    chain_count = 8,
    leap_size = 5,
    parallel = true,
) = TuringBI(sampler, warmup, samples_in_chain, chain_count, leap_size, parallel)

total_samples(t::TuringBI) = t.chain_count * t.samples_in_chain

function estimate_parameters(turing::TuringBI, problem::BossProblem, options::BossOptions)
    samples = sample_params(turing, problem.model, problem.data.X, problem.data.Y)
    return reduce_param_samples(samples...), nothing
end

Turing.@model function turing_model(
    model::Parametric,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    θ ~ product_distribution(model.theta_priors)
    noise_std ~ product_distribution(model.noise_std_priors)

    means = model.(eachcol(X), Ref(θ))
    
    Y ~ product_distribution(Distributions.MvNormal.(means, Ref(noise_std)))
end
Turing.@model function turing_model(
    model::GaussianProcess,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    y_dim = size(Y)[1]

    length_scales ~ product_distribution(model.length_scale_priors)
    amplitudes ~ product_distribution(model.amp_priors)
    noise_std ~ product_distribution(model.noise_std_priors)
    
    gps = [finite_gp(X, model.mean, model.kernel, length_scales[:,i], amplitudes[i], noise_std[i]) for i in 1:y_dim]

    Yt = transpose(Y)
    Yt ~ product_distribution(gps)
end
Turing.@model function turing_model(
    model::Semiparametric,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    y_dim = size(Y)[1]

    θ ~ product_distribution(model.parametric.theta_priors)
    length_scales ~ product_distribution(model.nonparametric.length_scale_priors)
    amplitudes ~ product_distribution(model.nonparametric.amp_priors)
    noise_std ~ product_distribution(model.nonparametric.noise_std_priors)

    mean = model.parametric(θ)
    gps = [finite_gp(X, x->mean(x)[i], model.nonparametric.kernel, length_scales[:,i], amplitudes[i], noise_std[i]) for i in 1:y_dim]
    
    Yt = transpose(Y)
    Yt ~ product_distribution(gps)
end

function sample_params(
    turing::TuringBI,
    model::Parametric,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    x_dim, y_dim = size(X)[1], size(Y)[1]
    θ_len, _, _, _ = param_counts(model)

    tm = turing_model(model, X, Y)
    chains = sample_params_turing(turing, tm)

    thetas = get_samples(chains, "θ", (θ_len,))
    length_scales = fill(nothing, total_samples(turing))
    amplitudes = fill(nothing, total_samples(turing))
    noise_std = get_samples(chains, "noise_std", (y_dim,))

    return thetas, length_scales, amplitudes, noise_std
end
function sample_params(
    turing::TuringBI,
    model::GaussianProcess,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    x_dim, y_dim = size(X)[1], size(Y)[1]
    
    tm = turing_model(model, X, Y)
    chains = sample_params_turing(turing, tm)
    
    thetas = fill(Float64[], total_samples(turing))
    length_scales = get_samples(chains, "length_scales", (x_dim, y_dim))
    amplitudes = get_samples(chains, "amplitudes", (y_dim,))
    noise_std = get_samples(chains, "noise_std", (y_dim,))

    return thetas, length_scales, amplitudes, noise_std
end
function sample_params(
    turing::TuringBI,
    model::Semiparametric,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)    
    x_dim, y_dim = size(X)[1], size(Y)[1]
    θ_len, _, _, _ = param_counts(model)
    
    tm = turing_model(model, X, Y)
    chains = sample_params_turing(turing, tm)

    thetas = get_samples(chains, "θ", (θ_len,))
    length_scales = get_samples(chains, "length_scales", (x_dim, y_dim))
    amplitudes = get_samples(chains, "amplitudes", (y_dim,))
    noise_std = get_samples(chains, "noise_std", (y_dim,))

    return thetas, length_scales, amplitudes, noise_std
end

function sample_params_turing(turing::TuringBI, turing_model)
    samples_in_chain = turing.warmup + (turing.leap_size * turing.samples_in_chain)
    if turing.parallel
        chains = Turing.sample(turing_model, turing.sampler, MCMCThreads(), samples_in_chain, turing.chain_count; progress=false)
    else
        chains = mapreduce(_ -> Turing.sample(turing_model, turing.sampler, samples_in_chain; progress=false), chainscat, 1:turing.chain_count)
    end
    chains = chains[turing.warmup+turing.leap_size:turing.leap_size:end]
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
