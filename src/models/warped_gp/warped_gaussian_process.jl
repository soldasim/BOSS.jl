
"""
    WarpedGaussianProcess(; kwargs...)

A Gaussian Process surrogate with an **adaptive parametric output transformation** (a warped GP).

Each output dimension `i` is modeled by an independent GP placed on the *latent* output
`δ = φᵢ(y)`, where `φᵢ` is a parametric [`OutputWarping`](@ref). The warping parameters are
fitted **jointly** with the GP hyperparameters by maximizing the GP marginal likelihood
including the Jacobian correction `Σⱼ log|φᵢ'(yⱼ)|`.

This differs fundamentally from [`TransformedModel`](@ref), whose transformation is fixed
(non-parametric) and applied as a preprocessing step without a Jacobian correction.

## Prediction

The latent GP predictive is Gaussian, `δ ~ N(m, s²)`. Predictions in observation space are
obtained via the analytical inverse warping `φ⁻¹` (Rios & Tobar, 2019):
- The **median** is `φ⁻¹(m)` (closed form).
- The **mean** `E[y]` and **variance** `Var[y]` are approximated by Gauss-Hermite quadrature:
```
E[y]   ≈ (1/√π) Σₖ wₖ φ⁻¹(m + √2 s zₖ)
Var[y] ≈ (1/√π) Σₖ wₖ φ⁻¹(m + √2 s zₖ)² - E[y]²
```
Note that under a nonlinear warping the observation-space posterior is non-Gaussian (skewed),
so the returned variance is an approximate summary of a skewed distribution.

## Keywords
- `mean::Union{Nothing, AbstractVector{<:Real}, Function}`: The GP mean function (in latent space).
        Defaults to `nothing`, equivalent to `x -> zeros(y_dim)`.
- `kernel::Kernel`: The GP kernel. Defaults to `Matern52Kernel()`.
- `lengthscale_priors::LengthscalePriors`: Priors for the GP length scales
        (a vector of `y_dim` `x_dim`-variate distributions). **Required** — no default.
- `amplitude_priors::AmplitudePriors`: Priors for the GP amplitudes (`y_dim` univariate distributions).
        Defaults to `[Dirac(1.0), ...]` (fixed amplitude = 1 per dimension). Since the default
        `output_warpings` end with an [`AffineWarping`](@ref), the affine scale parameter serves
        as the effective amplitude, making a learned GP amplitude redundant.
- `noise_std_priors::NoiseStdPriors`: Priors for the noise standard deviations (`y_dim` univariate
        distributions). Defaults to `[Dirac(0.0), ...]` (fixed noise = 0 per dimension).
- `output_warpings::AbstractVector{<:OutputWarping}`: One [`OutputWarping`](@ref) per output
        dimension. Defaults to `[ComposedWarping(YeoJohnsonWarping(), AffineWarping()), ...]`.
        The trailing [`AffineWarping`](@ref) captures the effective output mean and amplitude,
        so the GP mean and amplitude can be held fixed at 0 and 1.
- `quad_nodes::Int`: Number of Gauss-Hermite quadrature nodes used for prediction. Defaults to `20`.

## See Also

[`OutputWarping`](@ref), [`GaussianProcess`](@ref)
"""
struct WarpedGaussianProcess{
    M<:Union{Nothing, AbstractVector{<:Real}, Function},
    W<:AbstractVector{<:OutputWarping},
} <: SurrogateModel
    mean::M
    kernel::Kernel
    lengthscale_priors::LengthscalePriors
    amplitude_priors::AmplitudePriors
    noise_std_priors::NoiseStdPriors
    output_warpings::W
    quad_nodes::Int
end

function WarpedGaussianProcess(;
    mean = nothing,
    kernel = Matern52Kernel(),
    lengthscale_priors,
    amplitude_priors = nothing,
    noise_std_priors = nothing,
    output_warpings = nothing,
    quad_nodes = 20,
)
    mean_provided      = !isnothing(mean)
    amplitude_provided = !isnothing(amplitude_priors)

    y_dim = length(lengthscale_priors)
    isnothing(amplitude_priors) && (amplitude_priors = fill(Dirac(1.0), y_dim))
    isnothing(noise_std_priors) && (noise_std_priors = fill(Dirac(0.0), y_dim))
    isnothing(output_warpings) && (output_warpings =
        [ComposedWarping(YeoJohnsonWarping(), AffineWarping()) for _ in 1:y_dim])

    if mean_provided || amplitude_provided
        @warn """WarpedGaussianProcess: the GP mean and amplitude act in *latent* space \
(after the output warping), so they interact non-trivially with the warping parameters \
and can create identifiability issues. Consider keeping `mean = nothing` (≡ 0) and \
`amplitude_priors = [Dirac(1.0), ...]` and letting the warping absorb any offset and scale
with `AffineWarping` as the last transformation.""" maxlog=1
    end

    if (mean_provided || amplitude_provided) && any(_ends_with_affine, output_warpings)
        @warn """WarpedGaussianProcess: the output warping ends with an `AffineWarping`, \
whose shift and scale already capture the effective output mean and amplitude. \
Providing a separate GP mean or amplitude prior may introduce redundant parameters.""" maxlog=1
    end

    return WarpedGaussianProcess(mean, kernel, lengthscale_priors, amplitude_priors,
        noise_std_priors, output_warpings, quad_nodes)
end

"""
    WarpedGaussianProcessParams(λ, α, σ, warp)

The parameters of the [`WarpedGaussianProcess`](@ref) model.

## Parameters
- `λ::AbstractMatrix{<:Real}`: The GP length scales, shape `(x_dim, y_dim)`.
- `α::AbstractVector{<:Real}`: The GP amplitudes, length `y_dim`.
- `σ::AbstractVector{<:Real}`: The noise standard deviations, length `y_dim`.
- `warp::AbstractVector{<:AbstractVector{<:Real}}`: The warping parameters; `warp[i]` is the
        parameter vector `θ` of `output_warpings[i]`.
"""
struct WarpedGaussianProcessParams{
    L<:AbstractMatrix{<:Real},
    A<:AbstractVector{<:Real},
    N<:AbstractVector{<:Real},
    W<:AbstractVector{<:AbstractVector{<:Real}},
} <: ModelParams{WarpedGaussianProcess}
    λ::L
    α::A
    σ::N
    warp::W
end

make_discrete(m::WarpedGaussianProcess, discrete::AbstractVector{Bool}) =
    WarpedGaussianProcess(m.mean, make_discrete(m.kernel, discrete), m.lengthscale_priors,
        m.amplitude_priors, m.noise_std_priors, m.output_warpings, m.quad_nodes)


### Sliceable model interface ###

sliceable(::WarpedGaussianProcess) = true

function slice(m::WarpedGaussianProcess, idx::Int)
    return WarpedGaussianProcess(
        mean_slice(m.mean, idx),
        m.kernel,
        m.lengthscale_priors[idx:idx],
        m.amplitude_priors[idx:idx],
        m.noise_std_priors[idx:idx],
        m.output_warpings[idx:idx],
        m.quad_nodes,
    )
end

function slice(p::WarpedGaussianProcessParams, idx::Int)
    return WarpedGaussianProcessParams(
        p.λ[:, idx:idx],
        p.α[idx:idx],
        p.σ[idx:idx],
        p.warp[idx:idx],
    )
end

function join_slices(ps::AbstractVector{<:WarpedGaussianProcessParams})
    return WarpedGaussianProcessParams(
        hcat(getfield.(ps, Ref(:λ))...),
        vcat(getfield.(ps, Ref(:α))...),
        vcat(getfield.(ps, Ref(:σ))...),
        reduce(vcat, getfield.(ps, Ref(:warp))),
    )
end


### Gauss-Hermite quadrature ###

"""
Compute the `n`-point Gauss-Hermite quadrature nodes and weights (physicists' convention,
weight `exp(-x²)`, `Σ wₖ = √π`) via the Golub-Welsch algorithm.
"""
function _gauss_hermite(n::Int)
    n == 1 && return ([0.0], [sqrt(π)])
    β = [sqrt(k / 2) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), β))
    nodes = E.values
    weights = sqrt(π) .* (E.vectors[1, :] .^ 2)
    return nodes, weights
end


### Posterior ###

"""
    WarpedGaussianProcessPosterior

Posterior slice for a single output dimension of a [`WarpedGaussianProcess`](@ref).

## Fields
- `post_gp::AbstractGPs.PosteriorGP`: The latent posterior GP (in warped space).
- `warping::OutputWarping`: The output warping for this dimension.
- `warp_params::AbstractVector{<:Real}`: The fitted warping parameters `θ` (concrete subtype is preserved for specialization).
- `nodes`, `weights`: The Gauss-Hermite quadrature nodes and weights.
"""
struct WarpedGaussianProcessPosterior{
    P<:AbstractGPs.PosteriorGP,
    W<:OutputWarping,
    T<:AbstractVector{<:Real},
    NT<:AbstractVector{<:Real},
    WT<:AbstractVector{<:Real},
} <: ModelPosteriorSlice{WarpedGaussianProcess}
    post_gp::P
    warping::W
    warp_params::T
    nodes::NT
    weights::WT
end

function model_posterior_slice(
    model::WarpedGaussianProcess,
    params::WarpedGaussianProcessParams,
    data::ExperimentData,
    slice::Int,
)
    w = model.output_warpings[slice]
    θ = params.warp[slice]

    # Warp the observations into latent space and condition the GP there.
    δ = warp_forward.(Ref(w), Ref(θ), data.Y[slice, :])
    fgp = finite_gp(
        data.X,
        mean_getindex(model.mean, slice),
        model.kernel,
        params.λ[:, slice],
        params.α[slice],
        params.σ[slice],
    )
    post_gp = AbstractGPs.posterior(fgp, δ)

    nodes, weights = _gauss_hermite(model.quad_nodes)
    return WarpedGaussianProcessPosterior(post_gp, w, θ, nodes, weights)
end

"""
Back-transform a latent Gaussian predictive `N(m, σ2)` to the observation-space mean and
variance via Gauss-Hermite quadrature over the analytical inverse warping.
"""
function _back_transform(post::WarpedGaussianProcessPosterior, m::Real, σ2::Real)
    s = sqrt(max(σ2, zero(σ2)))
    ys = warp_inverse.(Ref(post.warping), Ref(post.warp_params), m .+ (sqrt(2) * s) .* post.nodes)
    norm = inv(sqrt(π))
    Ey = norm * sum(post.weights .* ys)
    Ey2 = norm * sum(post.weights .* ys .^ 2)
    return Ey, max(Ey2 - Ey^2, zero(Ey))
end

"""
    median(post::WarpedGaussianProcessPosterior, x) -> ::Real

The posterior median in observation space, `φ⁻¹(m)` (closed form, where `m` is the latent mean).
"""
function median(post::WarpedGaussianProcessPosterior, x::AbstractVector{<:Real})
    m = post.post_gp(hcat(x); obsdim=2) |> mean |> first
    return warp_inverse(post.warping, post.warp_params, m)
end

function mean_and_var(post::WarpedGaussianProcessPosterior, x::AbstractVector{<:Real})
    m, σ2 = post.post_gp(hcat(x); obsdim=2) |> mean_and_var .|> first
    return _back_transform(post, m, σ2) # ::Tuple{<:Real, <:Real}
end
function mean_and_var(post::WarpedGaussianProcessPosterior, X::AbstractMatrix{<:Real})
    ms, σ2s = post.post_gp(X; obsdim=2) |> mean_and_var
    res = _back_transform.(Ref(post), ms, σ2s)
    return first.(res), last.(res) # ::Tuple{<:AbstractVector, <:AbstractVector}
end

mean(post::WarpedGaussianProcessPosterior, x::AbstractVector{<:Real}) = first(mean_and_var(post, x))
mean(post::WarpedGaussianProcessPosterior, X::AbstractMatrix{<:Real}) = first(mean_and_var(post, X))

var(post::WarpedGaussianProcessPosterior, x::AbstractVector{<:Real}) = last(mean_and_var(post, x))
var(post::WarpedGaussianProcessPosterior, X::AbstractMatrix{<:Real}) = last(mean_and_var(post, X))

# Covariance in observation space is not well-defined under a nonlinear warping.
# The diagonal uses the Gauss-Hermite variance (consistent with `var`); off-diagonal entries
# use a delta-method (linearization) approximation `Cov[yᵢ,yⱼ] ≈ φ⁻¹'(mᵢ) φ⁻¹'(mⱼ) Cov[δᵢ,δⱼ]`,
# with `φ⁻¹'(m) = 1 / φ'(φ⁻¹(m))`.
function cov(post::WarpedGaussianProcessPosterior, X::AbstractMatrix{<:Real})
    ms, Σ_ = post.post_gp(X; obsdim=2) |> mean_and_cov
    n = length(ms)

    y_med = warp_inverse.(Ref(post.warping), Ref(post.warp_params), ms)
    g_prime = [inv(exp(warp_logderiv(post.warping, post.warp_params, y_med[i]))) for i in 1:n]

    Σ = similar(Σ_)
    for i in 1:n, j in 1:n
        if i == j
            _, v = _back_transform(post, ms[i], Σ_[i, i])
            Σ[i, j] = v
        else
            Σ[i, j] = g_prime[i] * g_prime[j] * Σ_[i, j]
        end
    end
    return Σ
end

function mean_and_cov(post::WarpedGaussianProcessPosterior, X::AbstractMatrix{<:Real})
    return mean(post, X), cov(post, X) # ::Tuple{<:AbstractVector{<:Real}, <:AbstractMatrix{<:Real}}
end


### Parameter methods ###

function data_loglike(model::WarpedGaussianProcess, data::ExperimentData)
    y_dim_ = size(data.Y, 1)

    function ll_data(params::WarpedGaussianProcessParams)
        slice_lls = map(1:y_dim_) do i
            w = model.output_warpings[i]
            θ = params.warp[i]
            y_row = data.Y[i, :]

            δ = warp_forward.(Ref(w), Ref(θ), y_row)
            gp_ll = gp_data_loglike_slice(
                data.X,
                δ,
                mean_getindex(model.mean, i),
                model.kernel,
                params.λ[:, i],
                params.α[i],
                params.σ[i],
            )
            # Jacobian correction for the change of variables δ = φ(y).
            jac = sum(warp_logderiv.(Ref(w), Ref(θ), y_row))
            return gp_ll + jac
        end
        return sum(slice_lls)
    end
end

function params_loglike(model::WarpedGaussianProcess)
    function ll_params(params::WarpedGaussianProcessParams)
        ll_λ = sum(logpdf.(model.lengthscale_priors, eachcol(params.λ)))
        ll_α = sum(logpdf.(model.amplitude_priors, params.α))
        ll_σ = sum(logpdf.(model.noise_std_priors, params.σ))
        ll_w = sum(_warp_loglike.(model.output_warpings, params.warp))
        return ll_λ + ll_α + ll_σ + ll_w
    end
end

function _warp_loglike(w::OutputWarping, θ)
    priors = warp_param_priors(w)
    isempty(priors) && return zero(eltype(θ))
    return sum(logpdf.(priors, θ))
end

function _params_sampler(model::WarpedGaussianProcess)
    function sample(rng::AbstractRNG)
        λ = hcat(rand.(Ref(rng), model.lengthscale_priors)...)
        α = rand.(Ref(rng), model.amplitude_priors)
        σ = rand.(Ref(rng), model.noise_std_priors)
        warp = [rand.(Ref(rng), warp_param_priors(w)) for w in model.output_warpings]
        return WarpedGaussianProcessParams(λ, α, σ, warp)
    end
end

function vectorizer(model::WarpedGaussianProcess)
    is_dirac, dirac_vals = create_dirac_mask(param_priors(model))
    warp_counts = warp_param_count.(model.output_warpings)

    function vectorize(params::WarpedGaussianProcessParams)
        ps = vcat(
            vec(params.λ),
            params.α,
            params.σ,
            reduce(vcat, params.warp; init=eltype(params.α)[]),
        )
        return filter_diracs(ps, is_dirac)
    end

    function devectorize(params::WarpedGaussianProcessParams, ps::AbstractVector{<:Real})
        ps = insert_diracs(ps, is_dirac, dirac_vals)

        λ_len = length(params.λ)
        α_len = length(params.α)
        σ_len = length(params.σ)
        λ_shape = size(params.λ)

        λ = reshape(ps[1:λ_len], λ_shape)
        α = ps[λ_len+1 : λ_len+α_len]
        σ = ps[λ_len+α_len+1 : λ_len+α_len+σ_len]

        rest = ps[λ_len+α_len+σ_len+1 : end]
        warp = Vector{Vector{eltype(rest)}}()
        idx = 1
        for c in warp_counts
            push!(warp, rest[idx:idx+c-1])
            idx += c
        end

        return WarpedGaussianProcessParams(λ, α, σ, warp)
    end

    return vectorize, devectorize
end

function bijector(model::WarpedGaussianProcess)
    b = default_bijector(param_priors(model))
    b = simplify(b)
    return b
end

function param_priors(model::WarpedGaussianProcess)
    warp_priors = isempty(model.output_warpings) ?
        Distribution[] : reduce(vcat, warp_param_priors.(model.output_warpings))
    return vcat(
        model.lengthscale_priors,
        model.amplitude_priors,
        model.noise_std_priors,
        warp_priors,
    )
end
