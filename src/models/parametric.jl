"""
    Parametric{N}

An abstract type for parametric surrogate models.

The model function can be reconstructed using the following functions:
- `(::Parametric)() -> ((x, θ) -> y)`
- `(::Parametric)(θ::AbstractVector{<:Real}) -> (x -> y)`
- `(::Parametric)(x::AbstractVector{<:Real}, θ::AbstractVector{<:Real}) -> y`

The parametric type `N <: Union{Nothing, NoiseStdPriors}`
determines whether the model is deterministic or probabilistic.

A deterministic version of a `Parametric` model has `N = nothing`,
does not implement the `SurrogateModel` API and cannot be used as a standalone model.
It is mainly used as a part of the [`Semiparametric`](@ref) model.

A probabilistic version of a `Parametric` model has defined `noise_std_priors`,
implements the whole `SurrogateModel` API, and can be used as a standalone model.

See also: [`LinearModel`](@ref), [`NonlinearModel`](@ref)
"""
abstract type Parametric{
    N<:Union{Nothing, NoiseStdPriors},
} <: SurrogateModel end

function (model::Parametric)()
    f = _construct_model(model)
    return f
end
function (model::Parametric)(θ::AbstractVector{<:Real})
    f = _construct_model(model)
    return x -> f(x, θ)
end
function (model::Parametric)(x::AbstractVector{<:Real}, θ::AbstractVector{<:Real})
    f = _construct_model(model)
    return f(x, θ)
end

"""
    LinearModel(; kwargs...)

A parametric surrogate model linear in its parameters.

This model definition will provide better performance than the more general 'NonlinearModel' in the future.
This feature is not implemented yet so it is equivalent to using `NonlinearModel` for now.

The linear model is defined as
```
     ϕs = lift(x)
     y = [θs[i]' * ϕs[i] for i in 1:m]
```
where
```
     x = [x₁, ..., xₙ]
     y = [y₁, ..., yₘ]
     θs = [θ₁, ..., θₘ], θᵢ = [θᵢ₁, ..., θᵢₚ]
     ϕs = [ϕ₁, ..., ϕₘ], ϕᵢ = [ϕᵢ₁, ..., ϕᵢₚ]
```
and ``n, m, p ∈ R``.

# Keywords
- `lift::Function`: Defines the `lift` function `(::Vector{<:Real}) -> (::Vector{Vector{<:Real}})`
        according to the definition above.
- `theta_priors::ThetaPriors`: The prior distributions for
        the parameters `[θ₁₁, ..., θ₁ₚ, ..., θₘ₁, ..., θₘₚ]` according to the definition above.
- `discrete::Union{Nothing, AbstractVector{Bool}}`: A vector of booleans indicating
        which dimensions of `x` are discrete. If `discrete = nothing`, all dimensions are continuous.
        Defaults to `nothing`.
- `noise_std_priors::Union{Nothing, NoiseStdPriors}`:
        The prior distributions of the noise standard deviations of each `y` dimension.
        If the model is used by itself, the `noise_std_priors` must be defined.
        If the model is used as a part of the `Semiparametric` model, the `noise_std_priors`
        must be left undefined, as the evaluation noise is modeled by the `GaussianProcess` in that case.
"""
@kwdef struct LinearModel{
    N<:Union{Nothing, NoiseStdPriors},
} <: Parametric{N}
    lift::Function
    theta_priors::ThetaPriors
    discrete::Union{Nothing, AbstractVector{Bool}} = nothing
    noise_std_priors::N = nothing
end

"""
    NonlinearModel(; kwargs...)

A parametric surrogate model.

If your model is linear, you can use `LinearModel` which will provide better performance in the future. (Not yet implemented.)

Define the model as `y = predict(x, θ)` where `θ` are the model parameters.

# Keywords
- `predict::Function`: The `predict` function according to the definition above.
- `theta_priors::ThetaPriors`: The prior distributions for the model parameters.
        function during optimization. Defaults to `nothing` meaning all parameters are real-valued.
- `discrete::Union{Nothing, AbstractVector{Bool}}`: A vector of booleans indicating
        which dimensions of `x` are discrete. If `discrete = nothing`, all dimensions are continuous.
        Defaults to `nothing`.
- `noise_std_priors::Union{Nothing, NoiseStdPriors}`:
        The prior distributions of the noise standard deviations of each `y` dimension.
        If the model is used by itself, the `noise_std_priors` must be defined.
        If the model is used as a part of the `Semiparametric` model, the `noise_std_priors`
        must be left undefined, as the evaluation noise is modeled by the `GaussianProcess` in that case.
"""
@kwdef struct NonlinearModel{
    N<:Union{Nothing, NoiseStdPriors},
} <: Parametric{N}
    predict::Function
    theta_priors::ThetaPriors
    discrete::Union{Nothing, AbstractVector{Bool}} = nothing
    noise_std_priors::N = nothing
end

"""
    ParametricParams(θ, σ)

The parameters of the [`Parametric`](@ref) model.

# Parameters
- `θ::AbstractVector{<:Real}`: The parameters of the [`Parametric`](@ref) model.
- `σ::AbstractVector{<:Real}`: The noise standard deviations.
"""
struct ParametricParams{
    P <: AbstractVector{<:Real},
    N <: AbstractVector{<:Real},
} <: ModelParams{Parametric}
    θ::P
    σ::N
end

function Base.convert(::Type{NonlinearModel}, model::LinearModel)
    return NonlinearModel(
        model(),
        model.theta_priors,
        model.discrete,
        model.noise_std_priors,
    )
end

remove_noise_priors(m::LinearModel) =
    LinearModel(m.lift, m.theta_priors, m.discrete, nothing)
remove_noise_priors(m::NonlinearModel) =
    NonlinearModel(m.predict, m.theta_priors, m.discrete, nothing)

make_discrete(m::LinearModel, discrete::AbstractVector{Bool}) =
    LinearModel(m.lift, m.theta_priors, discrete, m.noise_std_priors)
make_discrete(m::NonlinearModel, discrete::AbstractVector{Bool}) =
    NonlinearModel(m.predict, m.theta_priors, discrete, m.noise_std_priors)

param_count(params::ParametricParams) = sum(param_lengths(params))
param_lengths(params::ParametricParams) = (length(params.θ), length(params.σ))
param_shapes(params::ParametricParams) = (size(params.θ), size(params.σ))

# function barrier to infer the types of the arguments
_construct_model(model::LinearModel) = _construct_linear_model(model.lift, model.discrete)
_construct_model(model::NonlinearModel) = _construct_nonlinear_model(model.predict, model.discrete)

function _construct_linear_model(lift, discrete)
    function f(x, θ)
        x = discrete_round(discrete, x)

        ϕs = lift(x)
        m = length(ϕs)

        ϕ_lens = length.(ϕs)
        θ_indices = vcat(0, cumsum(ϕ_lens))
        
        y = [θ[θ_indices[i]+1:θ_indices[i+1]]' * ϕs[i] for i in 1:m]
        return y
    end
end

function _construct_nonlinear_model(predict, discrete)
    function f(x, θ)
        x = discrete_round(discrete, x)
        y = predict(x, θ)
        return y
    end
end

model_posterior(model::Parametric{<:NoiseStdPriors}, params::ParametricParams, data::ExperimentData) =
    model_posterior(model, params)
model_posterior(model::Parametric{<:NoiseStdPriors}, params::ParametricParams) =
    model_posterior(model, params.θ, params.σ)

function model_posterior(
    model::Parametric{<:NoiseStdPriors},
    theta::AbstractVector{<:Real},
    noise_std::AbstractVector{<:Real},
)
    f = model(theta)

    function post(x::AbstractVector{<:Real})
        μs = f(x)
        σs = noise_std
        return μs, σs # ::Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}
    end
    function post(X::AbstractMatrix{<:Real})
        count = size(X)[2]
        μs = mapreduce(f, hcat, eachcol(X))'
        Σs = mapreduce(σ -> Diagonal(fill(σ^2, count)), (a,b) -> cat(a,b; dims=3), noise_std)
        return μs, Σs # ::Tuple{<:AbstractMatrix{<:Real}, <:AbstractArray{<:Real, 3}}
    end
    return post
end

function data_loglike(model::Parametric{<:NoiseStdPriors}, data::ExperimentData)
    function ll_data(params::ParametricParams)
        post = model_posterior(model, params)
        loglike = mapreduce((x, y) -> logpdf(mvnormal(post(x)...), y), +, eachcol(data.X), eachcol(data.Y))
        return loglike
    end
end

function params_loglike(model::Parametric{<:NoiseStdPriors})
    function ll_params(params::ParametricParams)
        ll_theta = sum(logpdf.(model.theta_priors, params.θ); init=0.)
        ll_noise = sum(logpdf.(model.noise_std_priors, params.σ))
        return ll_theta + ll_noise
    end
end

function _params_sampler(model::Parametric{<:NoiseStdPriors})
    function sample(rng::AbstractRNG)
        θ = rand.(Ref(rng), model.theta_priors)
        σ = rand.(Ref(rng), model.noise_std_priors)
        return ParametricParams(θ, σ)
    end
end

function vectorizer(model::Parametric{<:NoiseStdPriors})
    is_dirac, dirac_vals = create_dirac_mask(param_priors(model))

    function vectorize(params::ParametricParams)
        ps = vcat(
            params.θ,
            params.σ,
        )

        ps = filter_diracs(ps, is_dirac)
        return ps
    end
    
    function devectorize(params::ParametricParams, ps::AbstractVector{<:Real})
        ps = insert_diracs(ps, is_dirac, dirac_vals)
        
        θ_len, σ_len = param_lengths(params)

        θ = ps[1:θ_len]
        σ = ps[θ_len+1:end]
    
        return ParametricParams(θ, σ)
    end

    return vectorize, devectorize
end

function bijector(model::Parametric{<:NoiseStdPriors})
    priors = param_priors(model)
    return default_bijector(priors)
end

function param_priors(model::Parametric{<:NoiseStdPriors})
    return vcat(
        model.theta_priors,
        model.noise_std_priors,
    )
end
