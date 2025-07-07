
"""
The minimal value of length scales and noise deviation used for GPs to avoid numerical issues.
"""
const MIN_PARAM_VALUE = 1e-8

"""
The maximum tolerated negative variance predicted by the posterior GP.

Predicted posterior variance values within the interval `(-MAX_NEG_VAR, 0)` are clipped to `0`.
A `DomainError` is still if values below `-MAX_NEG_VAR` occur.
"""
const MAX_NEG_VAR = 1e-8

"""
    GaussianProcess(; kwargs...)

A Gaussian Process surrogate model. Each output dimension is modeled by a separate independent process.

# Keywords
- `mean::Union{Nothing, AbstractVector{<:Real}, Function}`: Used as the mean function for the GP.
        Defaults to `nothing` equivalent to `x -> zeros(y_dim)`.
- `kernel::Kernel`: The kernel used in the GP. Defaults to the `Matern32Kernel()`.
- `lengthscale_priors::LengthscalePriors`: The prior distributions
        for the length scales of the GP. The `lengthscale_priors` should be a vector
        of `y_dim` `x_dim`-variate distributions where `x_dim` and `y_dim` are
        the dimensions of the input and output of the model respectively.
- `amplitude_priors::AmplitudePriors`: The prior distributions
        for the amplitude hyperparameters of the GP. The `amplitude_priors` should be a vector
        of `y_dim` univariate distributions.
- `noise_std_priors::NoiseStdPriors`: The prior distributions
        of the noise standard deviations of each `y` dimension.
"""
struct GaussianProcess{
    M<:Union{Nothing, AbstractVector{<:Real}, Function},
} <: SurrogateModel
    mean::M
    kernel::Kernel
    lengthscale_priors::LengthscalePriors
    amplitude_priors::AmplitudePriors
    noise_std_priors::NoiseStdPriors
end
# Keyword constructor for `GaussianProcess` is defined in `src/deprecated.jl`.

"""
    Nonparametric

An alias for `GaussianProcess`.
"""
const Nonparametric = GaussianProcess

"""
    GaussianProcessParams(λ, α, σ)

The parameters of the [`GaussianProcess`](@ref) model.

# Parameters
- `λ::AbstractMatrix{<:Real}`: The length scales of the GP.
- `α::AbstractVector{<:Real}`: The amplitudes of the GP.
- `σ::AbstractVector{<:Real}`: The noise standard deviations.
"""
struct GaussianProcessParams{
    L<:AbstractMatrix{<:Real},
    A<:AbstractVector{<:Real},
    N<:AbstractVector{<:Real},
} <: ModelParams{GaussianProcess}
    λ::L
    α::A
    σ::N
end

add_mean(m::GaussianProcess{Nothing}, mean) =
    GaussianProcess(mean, m.kernel, m.lengthscale_priors, m.amplitude_priors, m.noise_std_priors)

remove_mean(m::GaussianProcess) =
    GaussianProcess(nothing, m.kernel,  m.lengthscale_priors, m.amplitude_priors, m.noise_std_priors)

make_discrete(m::GaussianProcess, discrete::AbstractVector{Bool}) =
    GaussianProcess(m.mean, make_discrete(m.kernel, discrete), m.lengthscale_priors, m.amplitude_priors, m.noise_std_priors)

param_count(params::GaussianProcessParams) = sum(param_lengths(params))
param_lengths(params::GaussianProcessParams) = (length(params.λ), length(params.α), length(params.σ))
param_shapes(params::GaussianProcessParams) = (size(params.λ), size(params.α), size(params.σ))

sliceable(::GaussianProcess) = true

function slice(m::GaussianProcess, idx::Int)
    return GaussianProcess(
        mean_slice(m.mean, idx),
        m.kernel,
        m.lengthscale_priors[idx:idx],
        m.amplitude_priors[idx:idx],
        m.noise_std_priors[idx:idx],
    )
end

mean_slice(mean::Nothing, idx::Int) = nothing
mean_slice(mean::AbstractVector{<:Real}, idx::Int) = mean[idx:idx]
mean_slice(mean::Function, idx::Int) = x -> @view mean(x)[idx:idx]

mean_getindex(mean::Nothing, idx::Int) = nothing
mean_getindex(mean::AbstractVector{<:Real}, idx::Int) = mean[idx]
mean_getindex(mean::Function, idx::Int) = x -> mean(x)[idx]

function slice(p::GaussianProcessParams, idx::Int)
    return GaussianProcessParams(
        p.λ[:,idx:idx],
        p.α[idx:idx],
        p.σ[idx:idx],
    )
end

function join_slices(ps::AbstractVector{<:GaussianProcessParams})
    return GaussianProcessParams(
        hcat(getfield.(ps, Ref(:λ))...),
        vcat(getfield.(ps, Ref(:α))...),
        vcat(getfield.(ps, Ref(:σ))...),
    )
end

"""
    GaussianProcessPosterior

# Fields
- `post_gp::AbstractGPs.PosteriorGP`: The posterior GP constructed via the AbstractGPs.jl library.
"""
@kwdef struct GaussianProcessPosterior{
    P<:AbstractGPs.PosteriorGP,
} <: ModelPosteriorSlice{GaussianProcess}
    post_gp::P
end

function model_posterior_slice(
    model::GaussianProcess,
    params::GaussianProcessParams,
    data::ExperimentData,
    slice::Int,
)
    post_gp = posterior_gp(model, params, data, slice)
    return GaussianProcessPosterior(post_gp)
end

function mean(post::GaussianProcessPosterior, x::AbstractVector{<:Real})
    μ = post.post_gp(hcat(x); obsdim=2) |> mean |> first
    return μ # ::Real
end
function mean(post::GaussianProcessPosterior, X::AbstractMatrix{<:Real})
    μs = post.post_gp(X; obsdim=2) |> mean
    return μs # ::AbstractVector{<:Real}
end

function var(post::GaussianProcessPosterior, x::AbstractVector{<:Real})
    σ2 = post.post_gp(hcat(x); obsdim=2) |> var |> first
    σ2 = _clip_var(σ2)
    return σ2 # ::Real
end
function var(post::GaussianProcessPosterior, X::AbstractMatrix{<:Real})
    σ2s = post.post_gp(X; obsdim=2) |> var
    σ2s .= _clip_var.(σ2s)
    return σ2s # ::AbstractVector{<:Real}
end

function cov(post::GaussianProcessPosterior, X::AbstractMatrix{<:Real})
    Σs = post.post_gp(X; obsdim=2) |> cov
    Σs[diagind(Σs)] .= _clip_var.(diag(Σs))
    return Σs # ::AbstractMatrix{<:Real}
end

function mean_and_var(post::GaussianProcessPosterior, x::AbstractVector{<:Real})
    μ, σ2 = post.post_gp(hcat(x); obsdim=2) |> mean_and_var .|> first
    σ2 = _clip_var(σ2)
    return μ, σ2 # ::Tuple{<:Real, <:Real}
end
function mean_and_var(post::GaussianProcessPosterior, X::AbstractMatrix{<:Real})
    μs, σ2s = post.post_gp(X; obsdim=2) |> mean_and_var
    σ2s .= _clip_var.(σ2s)
    return μs, σ2s # ::Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}
end

function mean_and_cov(post::GaussianProcessPosterior, X::AbstractMatrix{<:Real})
    μs, Σs = post.post_gp(X; obsdim=2) |> mean_and_cov
    Σs[diagind(Σs)] .= _clip_var.(diag(Σs))
    return μs, Σs # ::Tuple{<:AbstractVector{<:Real}, <:AbstractMatrix{<:Real}}
end

function _clip_var(var::Real;
    threshold = MAX_NEG_VAR,    
)
    (var >= zero(var)) && return var
    (var >= -threshold) && return zero(var)
    throw(DomainError(var,
        "The posterior GP predicted variance $var but only values above -$threshold are tolerated."
    ))
end

"""
Construct posterior GP for a given `y` dimension via the AbstractGPs.jl library.
"""
function posterior_gp(model::GaussianProcess, params::GaussianProcessParams, data::ExperimentData, slice::Int)     
    return AbstractGPs.posterior(
        finite_gp(
            data.X,
            mean_getindex(model.mean, slice),
            model.kernel,
            params.λ[:,slice],
            params.α[slice],
            params.σ[slice],
        ),
        data.Y[slice,:],
    )
end

"""
Construct finite GP via the AbstractGPs.jl library.
"""
function finite_gp(
    X::AbstractMatrix{<:Real},
    mean::Union{Nothing, Real, Function},
    kernel::Kernel,
    lengthscales::AbstractVector{<:Real},
    amplitude::Real,
    noise_std::Real;
    min_param_val::Real = MIN_PARAM_VALUE,
)
    # zero values are set to `min_param_val` anyway
    # but negative values signal some error
    @assert all(lengthscales .>= 0)
    @assert amplitude >= 0
    @assert noise_std >= 0

    # for numerical stability
    # lengthscales = max.(lengthscales, min_param_val)
    # amplitude = max(amplitude, min_param_val)
    # noise_std = max(noise_std, min_param_val)
    lengthscales = lengthscales .+ min_param_val
    amplitude = amplitude + min_param_val
    noise_std = noise_std + min_param_val

    kernel = (amplitude^2) * with_lengthscale(kernel, lengthscales)
    return _finite_gp(X, mean, kernel, noise_std)
end

_finite_gp(X::AbstractMatrix{<:Real}, mean::Nothing, kernel::Kernel, noise_std::Real) = GP(kernel)(X, noise_std^2; obsdim=2)
_finite_gp(X::AbstractMatrix{<:Real}, mean, kernel::Kernel, noise_std::Real) = GP(mean, kernel)(X, noise_std^2; obsdim=2)

function data_loglike(
    model::GaussianProcess,
    data::ExperimentData,
)
    y_dim_ = size(data.Y)[1]

    function ll_data(params::GaussianProcessParams)
        return gp_data_loglike_slice.(
            Ref(data.X),
            eachrow(data.Y),
            mean_getindex.(Ref(model.mean), 1:y_dim_),
            Ref(model.kernel),
            eachcol(params.λ),
            params.α,
            params.σ,
        ) |> sum
    end
end

function gp_data_loglike_slice(
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real},
    mean,
    kernel::Kernel,
    lengthscales::AbstractVector{<:Real},
    amplitude::Real,
    noise_std::Real,
)
    gp = finite_gp(X, mean, kernel, lengthscales, amplitude, noise_std)
    return logpdf(gp, y)
end

function params_loglike(model::GaussianProcess)
    function ll_params(params::GaussianProcessParams)
        ll_λ = sum(logpdf.(model.lengthscale_priors, eachcol(params.λ)))
        ll_α = sum(logpdf.(model.amplitude_priors, params.α))
        ll_noise = sum(logpdf.(model.noise_std_priors, params.σ))
        return ll_λ + ll_α + ll_noise
    end
end

function _params_sampler(model::GaussianProcess)
    function sample(rng::AbstractRNG)
        λ = hcat(rand.(Ref(rng), model.lengthscale_priors)...)
        α = rand.(Ref(rng), model.amplitude_priors)
        σ = rand.(Ref(rng), model.noise_std_priors)
        return GaussianProcessParams(λ, α, σ)
    end
end

function vectorizer(model::GaussianProcess)
    is_dirac, dirac_vals = create_dirac_mask(param_priors(model))

    function vectorize(params::GaussianProcessParams)
        ps = vcat(
            vec(params.λ),
            params.α,
            params.σ,
        )
        
        ps = filter_diracs(ps, is_dirac)
        return ps
    end

    function devectorize(params::GaussianProcessParams, ps::AbstractVector{<:Real})
        ps = insert_diracs(ps, is_dirac, dirac_vals)
        
        λ_len, α_len, n_len = param_lengths(params)
        λ_shape = size(params.λ)

        λ = reshape(ps[1:λ_len], λ_shape)
        α = ps[λ_len+1:λ_len+α_len]
        σ = ps[λ_len+α_len+1:end]
    
        return GaussianProcessParams(λ, α, σ)
    end

    return vectorize, devectorize
end

function bijector(model::GaussianProcess)
    priors = param_priors(model)
    return default_bijector(priors)
end

function param_priors(model::GaussianProcess)
    return vcat(
        model.lengthscale_priors,
        model.amplitude_priors,
        model.noise_std_priors,
    )
end
