
"""
    GPMean

An abstract type for the mean options of a [`GaussianProcess`](@ref).

Subtypes: [`ZeroMean`](@ref), [`ConstantMean`](@ref), [`FunctionMean`](@ref), [`ConstantMeanPrior`](@ref).
"""
abstract type GPMean end

"""
    ZeroMean()

The GP has a zero mean.
"""
struct ZeroMean <: GPMean end

"""
    ConstantMean(values)

The GP has a fixed constant mean `values[i]` for the `i`-th output dimension.
"""
struct ConstantMean{V<:AbstractVector{<:Real}} <: GPMean
    values::V
end

"""
    FunctionMean(f)

The GP has a fixed mean function `f(x) -> AbstractVector{<:Real}`
returning the means of all output dimensions.
"""
struct FunctionMean{F<:Function} <: GPMean
    f::F
end

"""
    ConstantMeanPrior(priors)

The GP has a constant mean `μ[i]` for the `i`-th output dimension,
which is fitted as a model hyperparameter with the prior `priors[i]`.
"""
struct ConstantMeanPrior{P<:AbstractVector{<:UnivariateDistribution}} <: GPMean
    priors::P
end

"""
    as_mean(mean) -> ::GPMean

Convert the legacy mean specifications (`nothing`, a vector, a function) to a [`GPMean`](@ref).
"""
as_mean(mean::GPMean) = mean
as_mean(::Nothing) = ZeroMean()
as_mean(mean::AbstractVector{<:Real}) = ConstantMean(mean)
as_mean(mean::Function) = FunctionMean(mean)

mean_slice(m::ZeroMean, idx::Int) = m
mean_slice(m::ConstantMean, idx::Int) = ConstantMean(m.values[idx:idx])
mean_slice(m::FunctionMean, idx::Int) = FunctionMean(x -> @view m.f(x)[idx:idx])
mean_slice(m::ConstantMeanPrior, idx::Int) = ConstantMeanPrior(m.priors[idx:idx])

"""
    resolve_mean(mean::GPMean, μ, idx::Int) -> ::Union{Nothing, Real, Function}

Return the mean of the `idx`-th output dimension in the form accepted by `finite_gp`.
`μ` are the fitted constant means (`nothing` if the mean has no hyperparameters).
"""
resolve_mean(::ZeroMean, μ, idx::Int) = nothing
resolve_mean(m::ConstantMean, μ, idx::Int) = m.values[idx]
resolve_mean(m::FunctionMean, μ, idx::Int) = x -> m.f(x)[idx]
resolve_mean(::ConstantMeanPrior, μ::AbstractVector{<:Real}, idx::Int) = μ[idx]

"""
    mean_priors(mean::GPMean) -> ::AbstractVector{<:Distribution}

Return the priors of the mean hyperparameters (empty if there are none).
"""
mean_priors(::GPMean) = Distribution[]
mean_priors(m::ConstantMeanPrior) = m.priors

mean_params_sampler(::GPMean) = (rng::AbstractRNG) -> nothing
mean_params_sampler(m::ConstantMeanPrior) = (rng::AbstractRNG) -> rand.(Ref(rng), m.priors)

mean_params_logprior(::GPMean, ::Nothing) = 0.
mean_params_logprior(m::ConstantMeanPrior, μ::AbstractVector{<:Real}) = sum(logpdf.(m.priors, μ))

mean_params_length(::Nothing) = 0
mean_params_length(μ::AbstractVector{<:Real}) = length(μ)

slice_mean_params(::Nothing, idx::Int) = nothing
slice_mean_params(μ::AbstractVector{<:Real}, idx::Int) = μ[idx:idx]

join_mean_params(μs::AbstractVector{Nothing}) = nothing
join_mean_params(μs::AbstractVector{<:AbstractVector{<:Real}}) = reduce(vcat, μs)
