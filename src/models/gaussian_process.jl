
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
- `mean::Union{Nothing, Function}`: Used as the mean function for the GP.
        Defaults to `nothing` equivalent to `x -> zeros(y_dim)`.
- `kernel::Kernel`: The kernel used in the GP. Defaults to the `Matern32Kernel()`.
- `length_scale_priors::LengthScalePriors`: The prior distributions
        for the length scales of the GP. The `length_scale_priors` should be a vector
        of `y_dim` `x_dim`-variate distributions where `x_dim` and `y_dim` are
        the dimensions of the input and output of the model respectively.
- `amp_priors::AmplitudePriors`: The prior distributions
        for the amplitude hyperparameters of the GP. The `amp_priors` should be a vector
        of `y_dim` univariate distributions.
- `noise_std_priors::NoiseStdPriors`: The prior distributions
        of the noise standard deviations of each `y` dimension.
"""
@kwdef struct GaussianProcess{
    M<:Union{Nothing, Function},
    A<:AmplitudePriors,
    L<:LengthScalePriors,
    N<:NoiseStdPriors,
} <: SurrogateModel
    mean::M = nothing
    kernel::Kernel = Matern32Kernel()
    amp_priors::A
    length_scale_priors::L
    noise_std_priors::N
end

add_mean(m::GaussianProcess{Nothing}, mean::Function) =
    GaussianProcess(mean, m.kernel, m.amp_priors, m.length_scale_priors, m.noise_std_priors)

remove_mean(m::GaussianProcess) =
    GaussianProcess(nothing, m.kernel, m.amp_priors, m.length_scale_priors, m.noise_std_priors)

sliceable(::GaussianProcess) = true

function slice(m::GaussianProcess, idx::Int)
    mean_ = m.mean
    mean_slice_ = isnothing(mean_) ? nothing : x -> mean_(x)[idx:idx]

    return GaussianProcess(
        mean_slice_,
        m.kernel,
        m.amp_priors[idx:idx],
        m.length_scale_priors[idx:idx],
        m.noise_std_priors[idx:idx],
    )
end

mean_slice(mean::Nothing, idx::Int) = nothing
mean_slice(mean::Function, idx::Int) = x -> mean(x)[idx]

θ_slice(::GaussianProcess, ::Int) = nothing

make_discrete(m::GaussianProcess, discrete::AbstractVector{<:Bool}) =
    GaussianProcess(m.mean, make_discrete(m.kernel, discrete), m.amp_priors, m.length_scale_priors, m.noise_std_priors)

function model_posterior_slice(model::GaussianProcess, data::ExperimentDataMAP, slice::Int)
    post_gp = posterior_gp(model, data, slice)
    return model_posterior_slice(post_gp)
end
function model_posterior_slice(post_gp::AbstractGPs.PosteriorGP)    
    function post(x::AbstractVector{<:Real})
        mean_, var_ = mean_and_var(post_gp(hcat(x); obsdim=2)) .|> first
        var_ = _clip_var(var_)
        μ = mean_
        σ = sqrt(var_)
        return μ, σ # ::Tuple{<:Real, <:Real}
    end
    function post(X::AbstractMatrix{<:Real})
        μ, Σ = mean_and_cov(post_gp(X; obsdim=2))
        return μ, Σ # ::Tuple{<:AbstractVector{<:Real}, <:AbstractMatrix{<:Real}}
    end
    return post
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
function posterior_gp(model::GaussianProcess, data::ExperimentDataMAP, slice::Int) 
    _, length_scales, amplitudes, noise_std = data.params
    
    return AbstractGPs.posterior(
        finite_gp(
            data.X,
            mean_slice(model.mean, slice),
            model.kernel,
            length_scales[:,slice],
            amplitudes[slice],
            noise_std[slice],
        ),
        data.Y[slice,:],
    )
end

"""
Construct finite GP via the AbstractGPs.jl library.
"""
function finite_gp(
    X::AbstractMatrix{<:Real},
    mean::Union{Nothing, Function},
    kernel::Kernel,
    length_scales::AbstractVector{<:Real},
    amplitude::Real,
    noise_std::Real;
    min_param_val::Real = MIN_PARAM_VALUE,
)
    # for numerical stability
    length_scales = max.(length_scales, min_param_val)
    noise_std = max(noise_std, min_param_val)

    kernel = (amplitude^2) * with_lengthscale(kernel, length_scales)
    return finite_gp_(X, mean, kernel, noise_std)
end

finite_gp_(X::AbstractMatrix{<:Real}, mean::Nothing, kernel::Kernel, noise_std::Real) = GP(kernel)(X, noise_std^2; obsdim=2)
finite_gp_(X::AbstractMatrix{<:Real}, mean::Function, kernel::Kernel, noise_std::Real) = GP(mean, kernel)(X, noise_std^2; obsdim=2)

function model_loglike(model::GaussianProcess, data::ExperimentData)
    function loglike(params)
        ll_data = data_loglike(model, data.X, data.Y, params)
        ll_params = model_params_loglike(model, params)
        return ll_data + ll_params
    end
end

function data_loglike(
    model::GaussianProcess,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    params::ModelParams,
)
    y_dim = size(Y)[1]
    means = mean_slice.(Ref(model.mean), 1:y_dim)
    θ, λ, α, noise_std = params

    return sum(data_loglike_slice.(Ref(model), Ref(X), eachrow(Y), means, Ref(model.kernel), eachcol(λ), α, noise_std))
end

function data_loglike_slice(
    ::GaussianProcess,
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real},
    mean::Union{Nothing, Function},
    kernel::Kernel,
    length_scales::AbstractVector{<:Real},
    amplitude::Real,
    noise_std::Real,
)
    gp = finite_gp(X, mean, kernel, length_scales, amplitude, noise_std)
    return logpdf(gp, y)
end

function model_params_loglike(model::GaussianProcess, params::ModelParams)
    θ, λ, α, noise_std = params
    ll_λ = sum(logpdf.(model.length_scale_priors, eachcol(λ)))
    ll_α = sum(logpdf.(model.amp_priors, α))
    ll_noise = sum(logpdf.(model.noise_std_priors, noise_std))
    return ll_λ + ll_α + ll_noise
end

function sample_params(model::GaussianProcess)
    θ = nothing
    λ = reduce(hcat, rand.(model.length_scale_priors))
    α = rand.(model.amp_priors)
    noise_std = rand.(model.noise_std_priors)
    return θ, λ, α, noise_std
end

function param_priors(model::GaussianProcess)
    θ_priors = nothing
    λ_priors = model.length_scale_priors
    α_priors = model.amp_priors
    noise_std_priors = model.noise_std_priors
    return θ_priors, λ_priors, α_priors, noise_std_priors
end
