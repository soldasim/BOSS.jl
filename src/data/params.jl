
"""
An abstract type used to differentiate between
`MAP` (Maximum A Posteriori) and `BI` (Bayesian Inference) types.
"""
abstract type ModelFit end
struct MAP <: ModelFit end
struct BI <: ModelFit end

"""
    const Theta = AbstractVector{<:Real}

Parameters of the parametric model. Is empty in case of a nonparametric model.

Example: `[1., 2., 3.] isa Theta`
"""
const Theta = AbstractVector{<:Real}

"""
    const LengthScales = Union{Nothing, <:AbstractMatrix{<:Real}}

Length scales of the GP as a `x_dim`×`y_dim` matrix, or `nothing` if the model is purely parametric.

Example: `[1.;5.;; 1.;5.;;] isa LengthScales`
"""
const LengthScales = Union{Nothing, <:AbstractMatrix{<:Real}}

"""
    const Amplitudes = Union{Nothing, <:AbstractVector{<:Real}}

Amplitudes of the GP, or `nothing` if the model is purely parametric.

Example: `[1., 5.] isa Amplitudes`
"""
const Amplitudes = Union{Nothing, <:AbstractVector{<:Real}}

"""
    const NoiseStd = AbstractVector{<:Real}

Noise standard deviations of each `y` dimension.

Example: `[0.1, 1.] isa NoiseStd`
"""
const NoiseStd = AbstractVector{<:Real}

"""
    const ModelParams = Tuple{
        <:Theta,
        <:LengthScales,
        <:Amplitudes,
        <:NoiseStd,
    }

Represents all model (hyper)parameters.

Example:
```
params = (nothing, [1.;π;; 1.;π;;], [1., 1.5], [0.1, 1.])
params isa BOSS.ModelParams

θ, λ, α, noise = params
θ isa BOSS.Theta
λ isa BOSS.LengthScales
α isa BOSS.Amplitudes
noise isa BOSS.NoiseStd
```

See: [`Theta`](@ref), [`LengthScales`](@ref), [`Amplitudes`](@ref), [`NoiseStd`](@ref)
"""
const ModelParams = Tuple{
    <:Theta,
    <:LengthScales,
    <:Amplitudes,
    <:NoiseStd,
}

"""
    const ThetaPriors = AbstractVector{<:UnivariateDistribution}

Prior of [`Theta`](@ref).
"""
const ThetaPriors = AbstractVector{<:UnivariateDistribution}

"""
    const LengthScalePriors = Union{Nothing, <:AbstractVector{<:MultivariateDistribution}}

Prior of [`LengthScales`](@ref).
"""
const LengthScalePriors = Union{Nothing, <:AbstractVector{<:MultivariateDistribution}}

"""
    const AmplitudePriors = Union{Nothing, <:AbstractVector{<:UnivariateDistribution}}

Prior of [`Amplitudes`](@ref).
"""
const AmplitudePriors = Union{Nothing, <:AbstractVector{<:UnivariateDistribution}}

"""
    const NoiseStdPriors = AbstractVector{<:UnivariateDistribution}

Prior of [`NoiseStd`](@ref).
"""
const NoiseStdPriors = AbstractVector{<:UnivariateDistribution}

"""
    const ParamPriors = Tuple{
        <:ThetaPriors,
        <:LengthScalePriors,
        <:AmplitudePriors,
        <:NoiseStdPriors,
    }

Represents all (hyper)parameter priors.

See: [`ThetaPriors`](@ref), [`LengthScalePriors`](@ref), [`AmplitudePriors`](@ref), [`NoiseStdPriors`](@ref)
"""
const ParamPriors = Tuple{
    <:ThetaPriors,
    <:LengthScalePriors,
    <:AmplitudePriors,
    <:NoiseStdPriors,
}

function slice(params::ModelParams, θ_slice, idx::Int)
    θ, λ, α, noise_std = params
    θ_ = slice(θ, θ_slice)
    λ_ = slice(λ, idx)
    α_ = slice(α, idx)
    noise_std_ = slice(noise_std, idx)
    params_ = θ_, λ_, α_, noise_std_
    return params_
end

slice(M::AbstractMatrix, idx::Int) = M[:,idx:idx]
slice(v::AbstractVector, idx::Int) = v[idx:idx]

slice(v::AbstractVector, slice::Nothing) = empty(v)
slice(v::AbstractVector, slice::UnitRange{<:Int}) = v[slice]
