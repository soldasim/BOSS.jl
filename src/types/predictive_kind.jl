
"""
    PredictiveKind

Abstract supertype for the trait returned by [`predictive_kind`](@ref).

See also: [`GaussianPredictive`](@ref), [`SampledPredictive`](@ref).
"""
abstract type PredictiveKind end

"""
    GaussianPredictive <: PredictiveKind

Indicates that the model's posterior predictive distribution is (assumed) Gaussian, so its
`mean`/`var`/`mean_and_var` are the exact/complete description of the predictive.

This is the default returned by [`predictive_kind`](@ref).
"""
struct GaussianPredictive <: PredictiveKind end

"""
    SampledPredictive <: PredictiveKind

Indicates that, in addition to `mean`/`var`, the model provides a weighted-sample
discretization of its (generally non-Gaussian) predictive distribution via
[`predictive_samples`](@ref), which should be assumed **more precise** than the `mean`/`var`
summary.

Models whose predictive distribution genuinely is Gaussian should NOT return this trait —
see [`predictive_kind`](@ref).
"""
struct SampledPredictive <: PredictiveKind end

"""
    predictive_kind(::Type{<:SurrogateModel}) -> ::PredictiveKind
    predictive_kind(::SurrogateModel) -> ::PredictiveKind
    predictive_kind(::AbstractModelPosterior) -> ::PredictiveKind

Trait indicating whether the model's posterior predictive distribution is genuinely Gaussian
(`GaussianPredictive()`, the default — `mean`/`var`/`mean_and_var` are exact) or whether the
model additionally provides a more precise weighted-sample representation via
[`predictive_samples`](@ref) (`SampledPredictive()`).

Should be implemented **only** on the model's type, i.e. `predictive_kind(::Type{<:CustomModel}) = SampledPredictive()`.
The instance- and posterior-level methods (defined generically here and in
[`ModelPosterior`](@ref)/[`ModelPosteriorSlice`](@ref)) forward to this type-level method
automatically and should not be overridden separately — this keeps the trait's answer for a
model, its instances, and its posteriors always in sync by construction.

A model returning `SampledPredictive()` *should* implement [`predictive_samples`](@ref) on its
`ModelPosterior`/`ModelPosteriorSlice` type(s). A model whose predictive is genuinely Gaussian
(e.g. [`GaussianProcess`](@ref)) should leave this at the default `GaussianPredictive()` rather
than implementing a redundant, strictly noisier sample-based representation.

A wrapper model that reuses another model's predictive distribution unchanged (e.g.
[`TransformedModel`](@ref)) should carry that wrapped model's type as one of its own type
parameters and dispatch `predictive_kind` directly off it — see `TransformedModel` for the pattern.
"""
predictive_kind(::M) where {M<:SurrogateModel} = predictive_kind(M)
predictive_kind(::Type{<:SurrogateModel}) = GaussianPredictive()
