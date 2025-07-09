
"""
    Semiparametric(; kwargs...)

A semiparametric surrogate model (a combination of a `Parametric` model and a `GaussianProcess`).

The parametric model is used as the mean of the Gaussian Process
and the evaluation noise is modeled by the Gaussian Process.
All parameters of the models are estimated simultaneously.

## Keywords
- `parametric::Parametric`: The parametric model used as the GP mean function.
- `nonparametric::Nonparametric{Nothing}`: The outer GP model without mean.

Note that the parametric model should be defined without noise priors,
and the nonparametric model should be defined without mean function.
"""
@kwdef struct Semiparametric <: SurrogateModel
    parametric::Parametric{Nothing}         # parametric model without noise std priors
    nonparametric::Nonparametric{Nothing}   # nonparametric model without mean function

    function Semiparametric(parametric::Parametric, nonparametric::Nonparametric)
        _check_parametric_without_noise(parametric)
        _check_nonparametric_without_mean(nonparametric)
        parametric = remove_noise_priors(parametric)
        nonparametric = remove_mean(nonparametric)
        return new(parametric, nonparametric)
    end
end

function _check_parametric_without_noise(::Parametric{Nothing}) end
function _check_parametric_without_noise(::Parametric)
    @warn "The `noise_std_priors` of the `Parametric` model passed to the `Semiparametric` model are ignored."
end

function _check_nonparametric_without_mean(::Nonparametric{Nothing}) end
function _check_nonparametric_without_mean(::Nonparametric)
    @warn "The `mean` of the `Nonparametric` model passed to the `Semiparametric` model is ignored."
end

"""
    SemiparametricParams(θ, λ, α, σ)

The parameters of the [`Semiparametric`]@ref model.

## Parameters
- `θ::AbstractVector{<:Real}`: The parameters of the parametric model.
- `λ::AbstractMatrix{<:Real}`: The length scales of the GP.
- `α::AbstractVector{<:Real}`: The amplitudes of the GP.
- `σ::AbstractVector{<:Real}`: The noise standard deviations.
"""
struct SemiparametricParams{
    P <: AbstractVector{<:Real},
    L <: AbstractMatrix{<:Real},
    A <: AbstractVector{<:Real},
    N <: AbstractVector{<:Real},
} <: ModelParams{Semiparametric}
    θ::P
    λ::L
    α::A
    σ::N
end

function _extract_gp_params(params::SemiparametricParams)
    return GaussianProcessParams(
        params.λ,
        params.α,
        params.σ,
    )
end

make_discrete(m::Semiparametric, discrete::AbstractVector{Bool}) =
    Semiparametric(make_discrete(m.parametric, discrete), make_discrete(m.nonparametric, discrete))

param_count(params::SemiparametricParams) = sum(param_lengths(params))
param_lengths(params::SemiparametricParams) = (length(params.θ), length(params.λ), length(params.α), length(params.σ))
param_shapes(params::SemiparametricParams) = (size(params.θ), size(params.λ), size(params.α), size(params.σ))

function model_posterior_slice(model::Semiparametric, params::SemiparametricParams, data::ExperimentData, slice::Int)
    f = model.parametric(params.θ)
    gp = add_mean(model.nonparametric, f)
    post = model_posterior_slice(gp, _extract_gp_params(params), data, slice)
    return post # ::GaussianProcessPosterior
end

function data_loglike(model::Semiparametric, data::ExperimentData)
    function ll_data(params::SemiparametricParams)
        f = model.parametric(params.θ)
        gp = add_mean(model.nonparametric, f)
        return data_loglike(gp, data)(_extract_gp_params(params))
    end
end

function params_loglike(model::Semiparametric)
    function ll_params(params::SemiparametricParams)
        ll_theta = sum(logpdf.(model.parametric.theta_priors, params.θ); init=0.)
        ll_λ = sum(logpdf.(model.nonparametric.lengthscale_priors, eachcol(params.λ)))
        ll_α = sum(logpdf.(model.nonparametric.amplitude_priors, params.α))
        ll_noise = sum(logpdf.(model.nonparametric.noise_std_priors, params.σ))
        return ll_theta + ll_λ + ll_α + ll_noise
    end
end

function _params_sampler(model::Semiparametric)
    function sample(rng::AbstractRNG)
        θ = rand.(Ref(rng), model.parametric.theta_priors)
        λ = hcat(rand.(Ref(rng), model.nonparametric.lengthscale_priors)...)
        α = rand.(Ref(rng), model.nonparametric.amplitude_priors)
        σ = rand.(Ref(rng), model.nonparametric.noise_std_priors)
        return SemiparametricParams(θ, λ, α, σ)
    end
end

function vectorizer(model::Semiparametric)
    is_dirac, dirac_vals = create_dirac_mask(param_priors(model))

    function vectorize(params::SemiparametricParams)
        ps = vcat(
            params.θ,
            vec(params.λ),
            params.α,
            params.σ,
        )

        ps = filter_diracs(ps, is_dirac)
        return ps
    end

    function devectorize(params::SemiparametricParams, ps::AbstractVector{<:Real})
        ps = insert_diracs(ps, is_dirac, dirac_vals)
        
        θ_len, λ_len, α_len, n_len = param_lengths(params)
        λ_shape = size(params.λ)

        θ = ps[1:θ_len]
        λ = reshape(ps[θ_len+1:θ_len+λ_len], λ_shape)
        α = ps[θ_len+λ_len+1:θ_len+λ_len+α_len]
        σ = ps[θ_len+λ_len+α_len+1:end]

        return SemiparametricParams(θ, λ, α, σ)
    end

    return vectorize, devectorize
end

function bijector(model::Semiparametric)
    priors = param_priors(model)
    return default_bijector(priors)
end

function param_priors(model::Semiparametric)
    return vcat(
        model.parametric.theta_priors,
        model.nonparametric.lengthscale_priors,
        model.nonparametric.amplitude_priors,
        model.nonparametric.noise_std_priors,
    )
end
