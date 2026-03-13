
"""
    GradientGaussianProcess(; kwargs...)

A Gaussian Process surrogate conditioned on both function values and their gradients,
implementing the derivative-enhanced GP from Wu et al. (2017),
"Bayesian Optimization with Gradients".

Each simulator evaluation `(x, y, ∇y)` contributes `1 + x_dim` observations instead
of 1, improving sample efficiency. Use with [`GradientData`](@ref).

The simulator `f(x)` must return `(y, ∇y)` where:
- `y::Vector` is the function output of length `y_dim`
- `∇y::Vector` is the stacked row-wise Jacobian of length `y_dim * x_dim`:
  `∇y = vec(ForwardDiff.jacobian(f_y, x)')`

## Keywords

Same as `GaussianProcess`, plus:
- `grad_noise_std_priors::NoiseStdPriors`: Priors on gradient observation noise σ_∂.
  Should be non-Dirac to allow the GP to learn gradient uncertainty from data.
"""
@kwdef struct GradientGaussianProcess{
    M<:Union{Nothing, AbstractVector{<:Real}, Function},
} <: SurrogateModel
    mean::M = nothing
    kernel::Kernel = Matern52Kernel()
    lengthscale_priors::LengthscalePriors
    amplitude_priors::AmplitudePriors
    noise_std_priors::NoiseStdPriors
    grad_noise_std_priors::NoiseStdPriors
end

"""
    GradientGaussianProcessParams(λ, α, σ, σ_∂)

Parameters of [`GradientGaussianProcess`](@ref).

- `λ`: Lengthscales, shape `x_dim × y_dim`.
- `α`: Amplitudes, length `y_dim`.
- `σ`: Function observation noise std, length `y_dim`.
- `σ_∂`: Gradient observation noise std, length `y_dim`.
"""
struct GradientGaussianProcessParams{
    L<:AbstractMatrix{<:Real},
    A<:AbstractVector{<:Real},
    N<:AbstractVector{<:Real},
    ND<:AbstractVector{<:Real},
} <: ModelParams{GradientGaussianProcess}
    λ::L
    α::A
    σ::N
    σ_∂::ND
end

"""
Posterior slice for `GradientGaussianProcess`, holding precomputed quantities
for efficient prediction.
"""
struct GradientGPPosteriorSlice <: ModelPosteriorSlice{GradientGaussianProcess}
    k_fn::Any                    # (x, xp) -> scalar: the amplitude/lengthscale-scaled kernel
    X_train::Matrix{Float64}     # x_dim × n
    alpha::Vector{Float64}       # K_aug⁻¹ỹ, length n*(1 + x_dim)
    chol::Cholesky{Float64, Matrix{Float64}}
    σ::Float64                   # function observation noise std (stored for dKG)
    σ_∂::Float64                 # gradient observation noise std (stored for dKG)
end


### Sliceable model interface ###

sliceable(::GradientGaussianProcess) = true

function slice(m::GradientGaussianProcess, idx::Int)
    # Inline the mean-slice logic to avoid depending on BOSS internals.
    mean_idx = if isnothing(m.mean)
        nothing
    elseif m.mean isa AbstractVector
        m.mean[idx:idx]
    else
        x -> @view m.mean(x)[idx:idx]
    end
    return GradientGaussianProcess(
        mean_idx,
        m.kernel,
        m.lengthscale_priors[idx:idx],
        m.amplitude_priors[idx:idx],
        m.noise_std_priors[idx:idx],
        m.grad_noise_std_priors[idx:idx],
    )
end

function slice(p::GradientGaussianProcessParams, idx::Int)
    return GradientGaussianProcessParams(
        p.λ[:, idx:idx],
        p.α[idx:idx],
        p.σ[idx:idx],
        p.σ_∂[idx:idx],
    )
end

function join_slices(ps::AbstractVector{<:GradientGaussianProcessParams})
    return GradientGaussianProcessParams(
        hcat(getfield.(ps, Ref(:λ))...),
        vcat(getfield.(ps, Ref(:α))...),
        vcat(getfield.(ps, Ref(:σ))...),
        vcat(getfield.(ps, Ref(:σ_∂))...),
    )
end

param_lengths(p::GradientGaussianProcessParams) =
    (length(p.λ), length(p.α), length(p.σ), length(p.σ_∂))


### Kernel helpers ###

"""
    _make_kernel_fn(kernel, λ, α)

Build the scaled kernel function `(x, xp) -> k(x, xp)` for a given output slice,
with lengthscales and amplitude applied.
"""
function _make_kernel_fn(kernel::Kernel, lengthscales::AbstractVector, amplitude::Real)
    ε = MIN_PARAM_VALUE # const from gaussian_process.jl
    # Apply minimum threshold to ensure numerical stability
    α_scaled = (amplitude + ε)^2
    λ_scaled = lengthscales .+ ε
    return α_scaled * with_lengthscale(kernel, λ_scaled)
end

"""
    _kernel_and_derivs(k_fn, xi, xj)

Compute kernel value and derivatives at `(xi, xj)` for augmented GP observations.

Returns `(k_val, dk_dxi, dk_dxj, d2k)` where:
- `k_val`: Kernel value (scalar)
- `dk_dxi`: Gradient w.r.t. xi (length d)
- `dk_dxj`: Gradient w.r.t. xj (length d)  
- `d2k`: Hessian cross-derivative (d × d matrix)

To avoid NaN at diagonal where kernels have cusps (e.g., Matérn),
we perturb xj by ε when xi ≈ xj. Error is O(ε²) in Hessian.
"""
function _kernel_and_derivs(k_fn, xi::AbstractVector, xj::AbstractVector)
    d = length(xi)
    ε_perturb = MIN_PARAM_VALUE # TODO not pretty
    
    # Define function on concatenated input for joint differentiation
    f_combined(z) = k_fn(z[1:d], z[d+1:2d])
    z = vcat(xi, xj)
    
    # Perturb xj slightly if xi ≈ xj to avoid kernel singularities
    z_ad = xi ≈ xj ? vcat(xi, xj .+ ε_perturb) : z

    k_val = f_combined(z)
    grad = ForwardDiff.gradient(f_combined, z_ad)
    hess = ForwardDiff.hessian(f_combined, z_ad)
    
    return k_val, grad[1:d], grad[d+1:2d], hess[1:d, d+1:2d]
end

"""
Build the `N × N` augmented kernel matrix, N = n*(1 + x_dim).

Augmented observation ordering (consistent with `_build_obs_vector`):
  [f(x₁),...,f(xₙ),  ∂f/∂x₁(x₁),...,∂f/∂x₁(xₙ),  ...,  ∂f/∂x_d(x₁),...,∂f/∂x_d(xₙ)]

Block structure:
  K[i, j]                     = k(xᵢ, xⱼ)                   (function-function)
  K[i, n+(l-1)n+j]            = ∂k(xᵢ,xⱼ)/∂(xⱼ)_l           (function-gradient)
  K[n+(l-1)n+i, j]            = ∂k(xᵢ,xⱼ)/∂(xᵢ)_l           (gradient-function)
  K[n+(l-1)n+i, n+(m-1)n+j]  = ∂²k(xᵢ,xⱼ)/(∂(xᵢ)_l∂(xⱼ)_m) (gradient-gradient)

Noise terms: σ² on function block diagonal, σ_∂² on gradient block diagonal.
"""
function _build_augmented_kernel(k_fn, X::AbstractMatrix, σ::Real, σ_∂::Real)
    n = size(X, 2)
    d = size(X, 1)
    N = n * (1 + d)
    K = zeros(N, N)
    ε = MIN_PARAM_VALUE # const from gaussian_process.jl

    # Compute kernel matrix blocks
    for i in 1:n, j in 1:n
        k_val, dk_dxi, dk_dxj, d2k = _kernel_and_derivs(k_fn, X[:, i], X[:, j])
        
        # Function-function block: K[i, j] = k(xi, xj)
        K[i, j] = k_val
        
        # Function-gradient blocks
        for l in 1:d
            K[i, n + (l-1)*n + j] = dk_dxj[l]      # ∂k/∂xj_l
            K[n + (l-1)*n + i, j] = dk_dxi[l]      # ∂k/∂xi_l
        end
        
        # Gradient-gradient block
        for l in 1:d, m in 1:d
            K[n + (l-1)*n + i, n + (m-1)*n + j] = d2k[l, m]  # ∂²k/∂xi_l∂xj_m
        end
    end

    # Add noise to diagonal: σ² for function obs, σ_∂² for gradient obs
    noise_diag = vcat(
        fill((σ+ε)^2, n),          # Function observation noise
        fill((σ_∂+ε)^2, n * d),    # Gradient observation noise
    )
    K[diagind(K)] .+= noise_diag

    return Symmetric(K)
end

"""
    _build_cross_cov(k_fn, x_star, X_train)

Build augmented cross-covariance vector between test point `x_star`
and training observations (function values + gradients).

Length: n*(1 + d) with layout [f cov₁…f covₙ, ∂/∂x₁ cov₁…, …, ∂/∂xₐ covₙ].
"""
function _build_cross_cov(k_fn, x_star::AbstractVector, X_train::AbstractMatrix)
    n = size(X_train, 2)
    d = size(X_train, 1)
    ε_perturb = MIN_PARAM_VALUE # TODO not pretty
    
    k_cross = Vector{Float64}(undef, n * (1 + d))
    
    for j in 1:n
        xj = X_train[:, j]
        f_test(xp) = k_fn(x_star, xp)
        
        # Function value covariance
        k_cross[j] = f_test(xj)
        
        # Gradient covariance ∂k(x_star, xj)/∂xj_l
        xj_ad = x_star ≈ xj ? xj .+ ε_perturb : xj
        grad_k = ForwardDiff.gradient(f_test, xj_ad)
        
        for l in 1:d
            k_cross[n + (l-1)*n + j] = grad_k[l]
        end
    end
    
    return k_cross
end

"""
    _build_cross_cov_matrix(k_fn, x_new, X_train)

Build N × (1+d) matrix of covariances between training and new augmented observations.

Rows: [f obs₁…fobsₙ, ∂/∂x₁ obs₁…, …, ∂/∂xₐ obsₙ]  
Cols: [f(x_new), ∂f(x_new)/∂x₁, …, ∂f(x_new)/∂xₐ]
"""
function _build_cross_cov_matrix(k_fn, x_new::AbstractVector, X_train::AbstractMatrix)
    n = size(X_train, 2)
    d = length(x_new)
    N = n * (1 + d)
    K = zeros(N, 1 + d)
    
    for j in 1:n
        k_val, dk_dx_new, dk_dx_train, d2k = _kernel_and_derivs(k_fn, x_new, X_train[:, j])
        
        # Covariance with function value at x_new
        K[j, 1] = k_val
        
        # Covariances with gradients of x_new
        for l in 1:d
            K[j, 1+l] = dk_dx_new[l]  # ∂k/∂x_new_l
        end
        
        # Covariances with function value of training point
        for l in 1:d
            K[n+(l-1)*n+j, 1] = dk_dx_train[l]  # ∂k/∂x_train_l
        end
        
        # Covariances between training and new gradients
        for l in 1:d, m in 1:d
            K[n+(l-1)*n+j, 1+m] = d2k[m, l]  # ∂²k/∂x_train_l∂x_new_m
        end
    end
    
    return K
end

"""
    _build_obs_vector(y, dY)

Build augmented observation vector from function values and gradients.

Returns stacked vector [y₁,...yₙ, ∂y₁/∂x₁,...∂yₙ/∂x₁, ..., ∂y₁/∂xₐ,...∂yₙ/∂xₐ].

Args:
- `y`: Function values (length n)
- `dY`: Gradient matrix (d × n), where d = x_dim
"""
function _build_obs_vector(y::AbstractVector, dY::AbstractMatrix)
    d = size(dY, 1)
    # Stack gradients by dimension: [∂y/∂x₁, ∂y/∂x₂, ...]
    return vcat(y, [dY[l, :] for l in 1:d]...)
end


### Posterior construction ###

function model_posterior_slice(
    model::GradientGaussianProcess,
    params::GradientGaussianProcessParams,
    data::GradientData,
    slice::Int,
)
    # Extract parameters for this output slice
    k_fn = _make_kernel_fn(model.kernel, params.λ[:, slice], params.α[slice])
    σ = params.σ[slice]
    σ_∂ = params.σ_∂[slice]

    # Extract data for this output slice from the 3D Jacobian array
    X = data.X                    # x_dim × n
    y = data.Y[slice, :]          # Function values for this output slice
    dY = data.dY[slice, :, :]     # x_dim × n (Jacobian for this output)

    # Build augmented system and compute posterior
    ỹ = _build_obs_vector(y, dY)
    K_aug = _build_augmented_kernel(k_fn, X, σ, σ_∂)
    C = cholesky(K_aug)
    α_coeff = C \ ỹ

    return GradientGPPosteriorSlice(k_fn, Matrix(X), α_coeff, C, σ, σ_∂)
end


### Posterior prediction ###

function mean(post::GradientGPPosteriorSlice, x::AbstractVector{<:Real})
    k_cross = _build_cross_cov(post.k_fn, x, post.X_train)
    return k_cross ⋅ post.alpha
end

function mean(post::GradientGPPosteriorSlice, X::AbstractMatrix{<:Real})
    return [mean(post, X[:, j]) for j in axes(X, 2)]
end

function var(post::GradientGPPosteriorSlice, x::AbstractVector{<:Real})
    k_cross = _build_cross_cov(post.k_fn, x, post.X_train)
    k_self = post.k_fn(x, x)
    v = post.chol.L \ k_cross
    return max(0.0, k_self - v ⋅ v)
end

function var(post::GradientGPPosteriorSlice, X::AbstractMatrix{<:Real})
    return [var(post, X[:, j]) for j in axes(X, 2)]
end

function mean_and_var(post::GradientGPPosteriorSlice, x::AbstractVector{<:Real})
    k_cross = _build_cross_cov(post.k_fn, x, post.X_train)
    k_self = post.k_fn(x, x)
    μ = k_cross ⋅ post.alpha
    v = post.chol.L \ k_cross
    σ² = max(0.0, k_self - v ⋅ v)
    return μ, σ²
end

function mean_and_var(post::GradientGPPosteriorSlice, X::AbstractMatrix{<:Real})
    results = [mean_and_var(post, X[:, j]) for j in axes(X, 2)]
    return [r[1] for r in results], [r[2] for r in results]
end

function cov(post::GradientGPPosteriorSlice, X::AbstractMatrix{<:Real})
    cols = axes(X, 2)
    ks = [_build_cross_cov(post.k_fn, X[:, j], post.X_train) for j in cols]
    vs = [post.chol.L \ k for k in ks]
    return [post.k_fn(X[:, i], X[:, j]) - vs[i] ⋅ vs[j] for i in cols, j in cols]
end


### Data log-likelihood (log marginal likelihood of augmented GP) ###

function data_loglike(model::GradientGaussianProcess, data::GradientData)
    # Per-output log-likelihood for sliceable optimization by BOSS.jl
    function ll(params::GradientGaussianProcessParams)
        k_fn = _make_kernel_fn(model.kernel, params.λ[:, 1], params.α[1])
        σ = params.σ[1]
        σ_∂ = params.σ_∂[1]

        # Handle both unsliced (3D) and sliced (2D) data arrays
        if ndims(data.dY) == 3
            # Unsliced: extract first output from 3D array
            dY = data.dY[1, :, :]  # x_dim × n
            y = data.Y[1, :]
        else
            # Sliced: already 2D
            dY = data.dY  # x_dim × n
            y = data.Y[1, :]
        end

        ỹ = _build_obs_vector(y, dY)
        K_aug = _build_augmented_kernel(k_fn, data.X, σ, σ_∂)
        C = cholesky(K_aug)
        α_coeff = C \ ỹ
        N = length(ỹ)

        # Log marginal likelihood: -½(yỹ† K⁻¹ yỹ + log|K| + N log 2π)
        return -0.5 * (ỹ ⋅ α_coeff + 2 * sum(log.(diag(C.L))) + N * log(2π))
    end
    return ll
end


### Hyperparameter prior log-likelihood ###

function params_loglike(model::GradientGaussianProcess)
    function ll(params::GradientGaussianProcessParams)
        ll_λ  = sum(logpdf.(model.lengthscale_priors, eachcol(params.λ)))
        ll_α  = sum(logpdf.(model.amplitude_priors, params.α))
        ll_σ  = sum(logpdf.(model.noise_std_priors, params.σ))
        ll_σ_∂ = sum(logpdf.(model.grad_noise_std_priors, params.σ_∂))
        return ll_λ + ll_α + ll_σ + ll_σ_∂
    end
end

function BOSS._params_sampler(model::GradientGaussianProcess)
    function sample(rng::AbstractRNG)
        λ = hcat(rand.(Ref(rng), model.lengthscale_priors)...)
        α = rand.(Ref(rng), model.amplitude_priors)
        σ = rand.(Ref(rng), model.noise_std_priors)
        σ_∂ = rand.(Ref(rng), model.grad_noise_std_priors)
        return GradientGaussianProcessParams(λ, α, σ, σ_∂)
    end
end


### Vectorizer and bijector (for MAP optimization) ###

function vectorizer(model::GradientGaussianProcess)
    is_dirac, dirac_vals = BOSS.create_dirac_mask(param_priors(model))

    function vectorize(params::GradientGaussianProcessParams)
        ps = vcat(vec(params.λ), params.α, params.σ, params.σ_∂)
        return BOSS.filter_diracs(ps, is_dirac)
    end

    function devectorize(params::GradientGaussianProcessParams, ps::AbstractVector{<:Real})
        ps_full = BOSS.insert_diracs(ps, is_dirac, dirac_vals)
        λ_len, α_len, σ_len, σ_∂_len = param_lengths(params)
        
        # Unpack vectorized parameters back to structured form
        λ = reshape(ps_full[1:λ_len], size(params.λ))
        start_α = λ_len + 1
        end_α = start_α + α_len - 1
        α = ps_full[start_α:end_α]
        
        start_σ = end_α + 1
        end_σ = start_σ + σ_len - 1
        σ = ps_full[start_σ:end_σ]
        σ_∂ = ps_full[end_σ + 1:end]
        
        return GradientGaussianProcessParams(λ, α, σ, σ_∂)
    end

    return vectorize, devectorize
end

function bijector(model::GradientGaussianProcess)
    return BOSS.default_bijector(param_priors(model))
end

function param_priors(model::GradientGaussianProcess)
    return vcat(
        model.lengthscale_priors,
        model.amplitude_priors,
        model.noise_std_priors,
        model.grad_noise_std_priors,
    )
end
