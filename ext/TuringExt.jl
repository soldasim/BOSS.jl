module TuringExt

using BOSS
using Turing
using Distributions

"""
Implementation of the abstract `BOSS.TuringBI`. See the docs `? BOSS.TuringBI`.
"""
@kwdef struct TuringBI{S} <: BOSS.TuringBI
    sampler::S = PG(20)
    warmup::Int = 400
    samples_in_chain::Int = 20
    chain_count::Int = 4
    leap_size::Int = 5
    parallel::Bool = true
end

BOSS.TuringBI(args...; kwargs...) = TuringBI(args...; kwargs...)

total_samples(t::TuringBI) = t.chain_count * t.samples_in_chain

function BOSS.estimate_parameters(turing::TuringBI, problem::BossProblem, options::BossOptions)
    samples = sample_params(turing, problem.model, problem.data.X, problem.data.Y)
    return BOSS.reduce_param_samples(samples...), nothing
end

Turing.@model function turing_model(
    model::Parametric,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    θ ~ product_distribution(model.theta_priors)
    noise_std ~ product_distribution(model.noise_std_priors)

    means = model.(eachcol(X), Ref(θ))
    
    Y ~ product_distribution(MvNormal.(means, Ref(noise_std)))
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
    
    gps = [BOSS.finite_gp(X, BOSS.mean_slice(model.mean, i), model.kernel, length_scales[:,i], amplitudes[i], noise_std[i]) for i in 1:y_dim]

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
    gps = [BOSS.finite_gp(X, x->mean(x)[i], model.nonparametric.kernel, length_scales[:,i], amplitudes[i], noise_std[i]) for i in 1:y_dim]
    
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
    θ_len, _, _, _ = BOSS.param_counts(model)

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
    θ_len, _, _, _ = BOSS.param_counts(model)
    
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

end # module TuringExt
