using Turing
using Turing: Variational
using Zygote
using ForwardDiff
using Suppressor

"""
Stores hyperparameters of the Turing sampler.

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
struct TuringBI{S} <: ModelFitter{BI}
    sampler::S
    warmup::Int
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

sample_count(turing::TuringBI) = turing.chain_count * turing.samples_in_chain

function estimate_parameters(turing::TuringBI, problem::OptimizationProblem; info::Bool)
    θ, length_scales, noise_vars = sample_params(turing, problem.model, problem.noise_var_prior, problem.data.X, problem.data.Y; x_dim=x_dim(problem), y_dim=y_dim(problem))
    return (θ=θ, length_scales=length_scales, noise_vars=noise_vars)
end

Turing.@model function turing_model(model::Parametric, noise_var_prior, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}; y_dim::Int)
    noise_vars ~ noise_var_prior
    θ ~ arraydist(model.param_priors)

    means = model.(eachcol(X), Ref(θ))
    Y ~ arraydist(Distributions.MvNormal.(means, Ref(noise_vars)))
end
Turing.@model function turing_model(model::Nonparametric, noise_var_prior, X::AbstractMatrix{<:Real}, Y::AbstractVector{<:Real}; y_dim::Int)
    noise_vars ~ noise_var_prior
    length_scales ~ arraydist(model.length_scale_priors)
    
    gps = [finite_gp(X, nothing, kernel, length_scales[i], noise_vars[i]) for i in 1:y_dim]
    Y' ~ arraydist(gps)
end
Turing.@model function turing_model(model::Semiparametric, noise_var_prior, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}; y_dim::Int)
    noise_vars ~ noise_var_prior
    θ ~ arraydist(model.parametric.param_priors)
    length_scales ~ arraydist(model.nonparametric.length_scale_priors)

    mean = par_model(θ)
    gps = [finite_gp(X, x->mean(x)[i], kernel, length_scales[i], noise_vars[i]) for i in 1:y_dim]
    Y' ~ arraydist(gps)
end

function sample_params(turing::TuringBI, model::Parametric, noise_var_prior, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}; x_dim::Int, y_dim::Int)
    tm = turing_model(model, noise_var_prior, X, Y; y_dim)
    param_symbols = vcat(
        [Symbol("noise_vars[$i]") for i in 1:y_dim],
        [Symbol("θ[$i]") for i in 1:param_count(model)],
    )

    samples = sample_params_turing(turing, tm, param_symbols)
    noise_vars = reduce(vcat, transpose.(samples[1:y_dim]))
    θ = reduce(vcat, transpose.(samples[y_dim+1:end]))
    return θ, nothing, noise_vars
end
function sample_params(turing::TuringBI, model::Nonparametric, noise_var_prior, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}; x_dim::Int, y_dim::Int)    
    tm = turing_model(model, noise_var_prior, X, Y; y_dim)
    param_symbols = vcat(
        [Symbol("noise_vars[$i]") for i in 1:y_dim],
        reduce(vcat, [[Symbol("length_scales[$i][$j]") for j in 1:x_dim] for i in 1:y_dim]),
    )

    samples = sample_params_turing(turing, tm, param_symbols)
    noise_vars = reduce(vcat, transpose.(samples[1:y_dim]))
    length_scales = reshape.(eachcol(reduce(vcat, transpose.(samples[y_dim+1:end]))), Ref(x_dim), Ref(y_dim))
    return nothing, length_scales, noise_vars
end
function sample_params(turing::TuringBI, model::Semiparametric, noise_var_prior, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}; x_dim::Int, y_dim::Int)    
    θ_len = param_count(model.parametric)
    
    tm = turing_model(model, noise_var_prior, X, Y; y_dim)
    param_symbols = vcat(
        [Symbol("noise_vars[$i]") for i in 1:y_dim],
        [Symbol("θ[$i]") for i in 1:θ_len],
        reduce(vcat, [[Symbol("length_scales[$i][$j]") for j in 1:x_dim] for i in 1:y_dim]),
    )

    samples = sample_params_turing(turing, tm, param_symbols)
    noise_vars = reduce(vcat, transpose.(samples[1:y_dim]))
    θ = reduce(vcat, transpose.(samples[y_dim+1:y_dim+θ_len]))
    length_scales = reshape.(eachcol(reduce(vcat, transpose.(samples[y_dim+θ_len+1:end]))), Ref(x_dim), Ref(y_dim))
    return θ, length_scales, noise_vars
end

# Sample parameters of the given probabilistic model (defined with Turing.jl) using parallel NUTS MC sampling.
# Other AD backends than Zygote cause issues: https://discourse.julialang.org/t/gaussian-process-regression-with-turing-gets-stuck/86892
function sample_params_turing(turing::TuringBI, turing_model, param_symbols; adbackend=:zygote)
    Turing.setadbackend(adbackend)

    samples_in_chain = turing.warmup + (turing.leap_size * turing.samples_in_chain)
    if turing.parallel
        chains = Turing.sample(turing_model, turing.sampler, MCMCThreads(), samples_in_chain, turing.chain_count; progress=false)
    else
        chains = mapreduce(_ -> Turing.sample(turing_model, turing.sampler, samples_in_chain; progress=false), chainscat, 1:turing.chain_count)
    end

    samples = [reduce(vcat, eachrow(chains[s][(turing.warmup+turing.leap_size):turing.leap_size:end,:])) for s in param_symbols]
end








# TODO unused
function sample_params_vi(model, samples; alg=ADVI{Turing.AdvancedVI.ForwardDiffAD{0}}(10, 1000))
    posterior = vi(model, alg)
    rand(posterior, samples) |> eachrow |> collect
end
