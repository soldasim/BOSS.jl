
"""
    OutputWarping

Abstract supertype for **parametric, adaptive** output transformations used by
[`WarpedGaussianProcess`](@ref).

An output warping `φ` maps an observation `y` (observation space) to the latent space
`δ = φ(y)` on which a Gaussian Process is placed. The warping parameters `θ` are fitted
*jointly* with the GP hyperparameters by maximizing the GP marginal likelihood including
the Jacobian correction `Σᵢ log|φ'(yᵢ)|` (see [`WarpedGaussianProcess`](@ref)).

The warping is applied to a *single* output dimension. (The [`WarpedGaussianProcess`](@ref)
holds one `OutputWarping` per output dimension.)

## Defining a Custom Output Warping

Define a new `struct CustomWarping <: OutputWarping ... end` holding the **priors** of its
parameters (not the parameter values themselves), and implement:
- `warp_param_priors(::OutputWarping) -> ::AbstractVector{<:UnivariateDistribution}`:
        the parameter priors, in the order in which the parameters appear in `θ`.
- `warp_forward(::OutputWarping, θ, y::Real) -> δ::Real`:
        the forward warping `φ` (observation → latent).
- `warp_inverse(::OutputWarping, θ, δ::Real) -> y::Real`:
        the **closed-form** inverse `φ⁻¹` (latent → observation).
- `warp_logderiv(::OutputWarping, θ, y::Real) -> log|φ'(y)|::Real`:
        the log absolute derivative of the forward warping (for the Jacobian correction).

All warpings must be strictly monotone (so that `φ⁻¹` exists and is unique) and provide an
*analytical* inverse — this avoids the Newton–Raphson inner loop of Snelson's original
warped GP.

The compositional-warping design follows Rios & Tobar, *Compositionally-Warped Gaussian
Processes*, Neural Networks (2019), arXiv:1906.09665.

## See Also

[`AffineWarping`](@ref), [`YeoJohnsonWarping`](@ref),
[`SinhArcsinhWarping`](@ref), [`ComposedWarping`](@ref),
[`WarpedGaussianProcess`](@ref)
"""
abstract type OutputWarping end

"""
    warp_param_priors(::OutputWarping) -> ::AbstractVector{<:UnivariateDistribution}

Return the priors of the warping parameters, in the order in which they appear in `θ`.
"""
function warp_param_priors end

"""
    warp_forward(::OutputWarping, θ, y::Real) -> δ::Real

Apply the forward warping `δ = φ(y)` (observation space → latent space).
"""
function warp_forward end

"""
    warp_inverse(::OutputWarping, θ, δ::Real) -> y::Real

Apply the (closed-form) inverse warping `y = φ⁻¹(δ)` (latent space → observation space).
"""
function warp_inverse end

"""
    warp_logderiv(::OutputWarping, θ, y::Real) -> ::Real

Return the log absolute derivative `log|φ'(y)|` of the forward warping at `y`.
Used for the Jacobian correction of the data log-likelihood.
"""
function warp_logderiv end

"""
    warp_param_count(::OutputWarping) -> ::Int

Return the number of parameters of the warping (the length of `θ`).
"""
warp_param_count(w::OutputWarping) = length(warp_param_priors(w))


### AffineWarping ###

"""
    AffineWarping(; shift_prior, scale_prior)

The affine warping `φ(y) = shift + scale * y` with `scale > 0`.

Imposes no nonlinearity; useful as a final layer to rescale/shift after other warpings
(see [`ComposedWarping`](@ref)).

## Parameters
- `θ = [shift, scale]`, `scale > 0`.

## Keywords
- `shift_prior::UnivariateDistribution`: Prior on the shift parameter.
        Defaults to `Normal(0, 1)` (centered at the identity `shift = 0`).
- `scale_prior::UnivariateDistribution`: Prior on the (positive) scale parameter.
        Defaults to `truncated(Normal(1, 0.5); lower=0)` (centered at the identity `scale = 1`).
"""
@kwdef struct AffineWarping <: OutputWarping
    shift_prior::UnivariateDistribution = Normal(0., 1.)
    scale_prior::UnivariateDistribution = truncated(Normal(1., 0.5); lower=0.)
end

warp_param_priors(w::AffineWarping) = [w.shift_prior, w.scale_prior]

warp_forward(::AffineWarping, θ, y::Real) = θ[1] + θ[2] * y
warp_inverse(::AffineWarping, θ, δ::Real) = (δ - θ[1]) / θ[2]
warp_logderiv(::AffineWarping, θ, y::Real) = log(θ[2])


### YeoJohnsonWarping ###

"""
The threshold below which the Yeo-Johnson parameter `λ` (or `2 - λ`) is treated as zero
and the logarithmic limit of the transform is used.
"""
const YJ_TOL = 1e-6

"""
    YeoJohnsonWarping(; λ_prior)

The Yeo-Johnson warping (one parameter `λ`), defined on all of `ℝ`:
```
φ(y) =  ((y + 1)^λ - 1) / λ              for y ≥ 0, λ ≠ 0
        log(y + 1)                         for y ≥ 0, λ = 0
       -((-y + 1)^(2 - λ) - 1) / (2 - λ)  for y < 0, λ ≠ 2
       -log(-y + 1)                        for y < 0, λ = 2
```

Handles log-/power-scale behavior with no domain restriction (the recommended default warping).

## Finite-moment range

Although the transform itself is defined for all `λ ∈ ℝ`, its predictive *moments* (the mean
and variance computed by [`WarpedGaussianProcess`](@ref) via Gauss-Hermite quadrature) are
finite only for `λ ∈ [0, 2]`. Outside this range the transform is not surjective onto `ℝ`
(its image is bounded), so the inverse diverges and predictions far from the data become
`Inf`/`NaN`. The `λ_prior` should therefore be supported within `[0, 2]`; a warning is emitted
otherwise.

## Parameters
- `θ = [λ]`, `λ ∈ [0, 2]`. `λ = 1` is the identity.

## Keywords
- `λ_prior::UnivariateDistribution`: Prior on `λ`. Defaults to
        `truncated(Normal(1, 0.5); lower=0, upper=2)` (centered at the identity `λ = 1`,
        restricted to the finite-moment range).
"""
@kwdef struct YeoJohnsonWarping <: OutputWarping
    λ_prior::UnivariateDistribution = truncated(Normal(1., 0.5); lower=0., upper=2.)

    function YeoJohnsonWarping(λ_prior::UnivariateDistribution)
        if (minimum(λ_prior) < 0) || (maximum(λ_prior) > 2)
            @warn "`YeoJohnsonWarping`: the `λ_prior` has support outside [0, 2]. The Yeo-Johnson \
                predictive moments (mean/variance) diverge for λ ∉ [0, 2], which can yield \
                Inf/NaN predictions far from the data. Consider restricting the prior, e.g. \
                `truncated(...; lower=0, upper=2)`." maxlog=1
        end
        return new(λ_prior)
    end
end

warp_param_priors(w::YeoJohnsonWarping) = [w.λ_prior]

function warp_forward(::YeoJohnsonWarping, θ, y::Real)
    λ = θ[1]
    if y >= 0
        return abs(λ) < YJ_TOL ? log(y + 1) : ((y + 1)^λ - 1) / λ
    else
        return abs(λ - 2) < YJ_TOL ? -log(-y + 1) : -((-y + 1)^(2 - λ) - 1) / (2 - λ)
    end
end

function warp_inverse(::YeoJohnsonWarping, θ, δ::Real)
    λ = θ[1]
    # `δ ≥ 0 ⟺ y ≥ 0`, since `φ` is monotone with `φ(0) = 0`.
    # Clamp bases to 0 before exponentiation: floating-point can make them
    # negligibly negative even when the true value is 0.
    if δ >= 0
        return abs(λ) < YJ_TOL ? exp(δ) - 1 : max(λ * δ + 1, 0.0)^(1 / λ) - 1
    else
        return abs(λ - 2) < YJ_TOL ? 1 - exp(-δ) : 1 - max(1 - (2 - λ) * δ, 0.0)^(1 / (2 - λ))
    end
end

function warp_logderiv(::YeoJohnsonWarping, θ, y::Real)
    λ = θ[1]
    # φ'(y) = (y + 1)^(λ - 1) for y ≥ 0,  (-y + 1)^(1 - λ) for y < 0 (always > 0).
    return y >= 0 ? (λ - 1) * log(y + 1) : (1 - λ) * log(-y + 1)
end


### SinhArcsinhWarping ###

"""
    SinhArcsinhWarping(; skewness_prior, tailweight_prior)

The sinh-arcsinh warping `φ(y) = sinh(b · arcsinh(y) - a)` with `b > 0`.

The parameter `a` controls skewness and `b` controls tail weight / kurtosis;
`a = 0, b = 1` is the identity. Useful as an add-on layer to correct residual
skewness and tail heaviness (see [`ComposedWarping`](@ref)).

## Parameters
- `θ = [a, b]`, `b > 0`.

## Keywords
- `skewness_prior::UnivariateDistribution`: Prior on the skewness parameter `a`.
        Defaults to `Normal(0, 1)` (centered at the identity `a = 0`).
- `tailweight_prior::UnivariateDistribution`: Prior on the (positive) tail-weight parameter `b`.
        Defaults to `truncated(Normal(1, 0.5); lower=0)` (centered at the identity `b = 1`).
"""
@kwdef struct SinhArcsinhWarping <: OutputWarping
    skewness_prior::UnivariateDistribution = Normal(0., 1.)
    tailweight_prior::UnivariateDistribution = truncated(Normal(1., 0.5); lower=0.)
end

warp_param_priors(w::SinhArcsinhWarping) = [w.skewness_prior, w.tailweight_prior]

warp_forward(::SinhArcsinhWarping, θ, y::Real) = sinh(θ[2] * asinh(y) - θ[1])
warp_inverse(::SinhArcsinhWarping, θ, δ::Real) = sinh((asinh(δ) + θ[1]) / θ[2])

function warp_logderiv(::SinhArcsinhWarping, θ, y::Real)
    a, b = θ[1], θ[2]
    # φ'(y) = b · cosh(b·arcsinh(y) - a) / sqrt(1 + y²)
    return log(b) + log(cosh(b * asinh(y) - a)) - 0.5 * log(1 + y^2)
end


### ComposedWarping ###

"""
    ComposedWarping(warps::AbstractVector{<:OutputWarping})
    ComposedWarping(warps::OutputWarping...)

A warping formed by composing elementary warpings:
`φ = φ_d ∘ φ_{d-1} ∘ ⋯ ∘ φ_1`.

The forward warping applies `warps[1]` first, then `warps[2]`, and so on.
The inverse reverses the composition, and the log-derivative is accumulated via the chain rule.
All sub-warpings remain analytically invertible, so the composition is too.

The recommended general-purpose stack is `YeoJohnsonWarping → SinhArcsinhWarping`
(handles log-scale compression, then residual skewness/tails).

## Fields
- `warps::Vector{OutputWarping}`: The elementary warpings, in forward-application order.
"""
struct ComposedWarping <: OutputWarping
    warps::Vector{OutputWarping}
end
ComposedWarping(warps::OutputWarping...) = ComposedWarping(collect(OutputWarping, warps))

warp_param_priors(w::ComposedWarping) =
    isempty(w.warps) ? UnivariateDistribution[] : reduce(vcat, warp_param_priors.(w.warps))

_warp_param_counts(w::ComposedWarping) = warp_param_count.(w.warps)

# Split a flat parameter vector `θ` into per-sub-warping chunks.
function _split_params(w::ComposedWarping, θ)
    chunks = Vector{typeof(θ)}(undef, length(w.warps))
    i = 1
    for (k, c) in enumerate(_warp_param_counts(w))
        chunks[k] = θ[i:i+c-1]
        i += c
    end
    return chunks
end

function warp_forward(w::ComposedWarping, θ, y::Real)
    chunks = _split_params(w, θ)
    z = y
    for (wi, θi) in zip(w.warps, chunks)
        z = warp_forward(wi, θi, z)
    end
    return z
end

function warp_inverse(w::ComposedWarping, θ, δ::Real)
    chunks = _split_params(w, θ)
    z = δ
    for (wi, θi) in Iterators.reverse(collect(zip(w.warps, chunks)))
        z = warp_inverse(wi, θi, z)
    end
    return z
end

function warp_logderiv(w::ComposedWarping, θ, y::Real)
    chunks = _split_params(w, θ)
    z = y
    total = zero(warp_logderiv(first(w.warps), first(chunks), y))
    for (wi, θi) in zip(w.warps, chunks)
        total += warp_logderiv(wi, θi, z)
        z = warp_forward(wi, θi, z)
    end
    return total
end

"""Return `true` if the warping's outermost (last-applied) layer is an `AffineWarping`."""
_ends_with_affine(::AffineWarping) = true
_ends_with_affine(w::ComposedWarping) = !isempty(w.warps) && last(w.warps) isa AffineWarping
_ends_with_affine(::OutputWarping) = false
