
# --- Activation Functions ---

"""
    IBNNActivationFunction

The activation function of a `IBNNLayer`.

Activation function *should* implement:
- `F(::CustomActivationFunction, kxx::Real, kxy::Real, kyy::Real) -> ::Real`:
        The `F` function according to [1].

Activation function *may* implement:
- `(::CustomActivationFunction)(x::AbstractVector{<:Real}) -> ::AbstractVector{<:Real}`:
        The definition of the activation function (the forward pass). Not actually used anywhere.

# References
[1] Lee, Jaehoon, et al. "Deep neural networks as gaussian processes." arXiv preprint arXiv:1711.00165 (2017).
"""
abstract type IBNNActivationFunction end

"""
    F(::IBNNActivationFunction, kxx::Real, kxy::Real, kyy::Real) -> ::Real

The `F` function of a NN activation function from [1].

# References
[1] Lee, Jaehoon, et al. "Deep neural networks as gaussian processes." arXiv preprint arXiv:1711.00165 (2017).
"""
function F end

"""
    IBNNReLU()

The ReLU activation function for `IBNN`.
"""
struct IBNNReLU <: IBNNActivationFunction end

(::IBNNReLU)(x::AbstractVector{<:Real}) = max.(eltype(x)(0), x)

function F(::IBNNReLU, kxx::Real, kxy::Real, kyy::Real)
    k_ = sqrt(kxx * kyy)
    θ = safe_acos(kxy / k_)
    return (1 / 2π) * k_ * (sin(θ) + (π - θ) * cos(θ))
end

function safe_acos(x; ϵ=1e-6) # TODO ϵ
    # TODO
    if x < -1.
        if x < -1. - ϵ
            error("acos(x) is undefined for x < -1.")
        else
            return π
        end
    elseif x > 1.
        if x > 1. + ϵ
            error("acos(x) is undefined for x > 1.")
        else
            return 0.
        end
    else
        return acos(x)
    end

    # x = max(-1. + ϵ, x)
    # x = min(1. - ϵ, x)
    # return acos(x)
end


# --- The Surrogate Model ---

"""
    IBNNPriors(; kwargs...)

Defines all hyperparameter priors for `IBNN`.

The `omega_priors` define the priors for the `ω` parameters of the `IBNN` for each `y` dimension.
*Either* `omega_priors` *or* the `ω2_w0`, `ω2_b0`, `ω2_w`, `ω2_b` parameters
should be defined. Never both.

If `omega_priors` is defined, the `ω` parameters are treated as additional Bayesian hyperparameters,
where a single scalar value is used for all `ω` parameters of each `y` dimension.

Or the `ω` parameters can be defined as fixed values via the `ω2_w0`, `ω2_b0`, `ω2_w`, `ω2_b`
fields of `IBNN` and its layers.
"""
@kwdef struct IBNNPriors{
    S<:Union{Nothing, ThetaPriors},
    L<:LengthscalePriors,
    A<:AmplitudePriors,
    N<:NoiseStdPriors,
}
    omega_priors::S = nothing
    lengthscale_priors::L
    amplitude_priors::A
    noise_std_priors::N

    function IBNNPriors(
        omega_priors::S,
        lengthscale_priors::L,
        amplitude_priors::A,
        noise_std_priors::N,
    ) where {S,L,A,N}
        if !isnothing(omega_priors)
            @assert length(omega_priors) == length(amplitude_priors)
        end
        return new{S,L,A,N}(omega_priors, lengthscale_priors, amplitude_priors, noise_std_priors)
    end
end

"""
    IBNNLayer(; kwargs...)

A single layer of `IBNN`.
"""
@kwdef struct IBNNLayer{
    A<:IBNNActivationFunction,
    S<:Union{Nothing, Float64},
}
    act_func::A
    ω2_w::S = nothing # σ2_w = ω2_w / layer_width (where layer_width == inf)
    ω2_b::S = nothing # σ2_b = ω2_b / 1
end

"""
    IBNN(; kwargs...)

An infinite-width bayesian neural network according to [1].

Either define scalar values `ω2_w0`, `ω2_b0` in `IBNN` (the 0th layer) and `ω2_w`, `ω2_b` in each layer,
OR define `omega_priors` in `IBNNPriors` and the ω values will be treated as an additional Bayesian hyperparameter.

# References
[1] Lee, Jaehoon, et al. "Deep neural networks as gaussian processes." arXiv preprint arXiv:1711.00165 (2017).
"""
@kwdef struct IBNN{
    S<:Union{Nothing, Float64},
} <: SurrogateModel
    ω2_w0::S = nothing # σ2_w0 = ω2_w0 / x_dim
    ω2_b0::S = nothing # σ2_b0 = ω2_b0 / 1
    layers::Vector{<:IBNNLayer{<:IBNNActivationFunction, S}}
    priors::IBNNPriors

    function IBNN(
        ω2_w0::S,
        ω2_b0::S,
        layers::AbstractVector{<:IBNNLayer{<:IBNNActivationFunction, S}},
        priors::IBNNPriors,
    ) where {S}
        (length(layers) == 0) && error("The I-BNN must have at least one hidden layer.
            (Otherwise, the CLT would not apply and the output would not follow a Gaussian distribution.)")
        
        # "Sigma prior is defined." XOR "All sigmas are defined."
        if S <: Nothing
            (priors isa IBNNPriors{Nothing}) && error("Define `IBNN.priors.omega_priors` OR all NN parameter sigmas (`IBNN.ω2_w0`, `IBNN.ω2_b0`, `IBNN.layers[i].ω2_w`, `IBNN.layers[i].ω2_b`).")
        else
            (priors isa IBNNPriors{Nothing}) || error("`IBNN.priors.omega_priors` and NN parameters sigmas (`IBNN.ω2_w0`, `IBNN.ω2_b0`, `IBNN.layers[i].ω2_w`, `IBNN.layers[i].ω2_b`) cannot be both defined at the same time.")
        end
        
        return new{S}(ω2_w0, ω2_b0, layers, priors)
    end
end

"""
    IBNNParams(ω, λ, α, noise_std)

The parameters of `IBNN` surrogate model.

# Parameters
- `ω::Union{Nothing, AbstractVector{<:Real}}`: The `ω` parameters of the `IBNN`.
        If `ω == nothing`, the fixed values `ω2_w0`, `ω2_b0`, `ω2_w`, `ω2_b`
        defined in the `IBNN` and its `IBNNLayer`s are used instead.
- `λ::AbstractMatrix{<:Real}`: The lengthscale parameters of the `IBNN`.
- `α::AbstractVector{<:Real}`: The amplitude parameters of the `IBNN`.
- `noise_std::AbstractVector{<:Real}`: The noise standard deviations.
"""
struct IBNNParams{
    S <: Union{Nothing, AbstractVector{<:Real}},
    L <: AbstractMatrix{<:Real},
    A <: AbstractVector{<:Real},
    N <: AbstractVector{<:Real},
} <: ModelParams{IBNN}
    ω::S
    λ::L
    α::A
    noise_std::N
end

depth(model::IBNN) = length(model.layers)

# TODO unimplemented
# function make_discrete(model::IBNN, discrete::AbstractVector{Bool}) end

param_count(params::IBNNParams) = sum(param_lengths(params))
param_lengths(params::IBNNParams{Nothing}) = (length(params.λ), length(params.α), length(params.noise_std))
param_lengths(params::IBNNParams{<:AbstractVector{<:Real}}) = (length(params.ω), length(params.λ), length(params.α), length(params.noise_std))
param_shapes(params::IBNNParams{Nothing}) = (size(params.λ), size(params.α), size(params.noise_std))
param_shapes(params::IBNNParams{<:AbstractVector{<:Real}}) = (size(params.ω), size(params.λ), size(params.α), size(params.noise_std))

sliceable(::IBNN) = true

function slice(m::IBNN, idx::Int)
    return IBNN(
        m.ω2_w0,
        m.ω2_b0,
        m.layers,
        slice(m.priors, idx),
    )
end

function slice(priors::IBNNPriors, idx::Int)
    return IBNNPriors(
        isnothing(priors.omega_priors) ? nothing : priors.omega_priors[idx:idx],
        priors.lengthscale_priors[idx:idx],
        priors.amplitude_priors[idx:idx],
        priors.noise_std_priors[idx:idx],
    )
end

function slice(params::IBNNParams, idx::Int)
    return IBNNParams(
        isnothing(params.ω) ? nothing : params.ω[idx:idx],
        params.λ[:,idx:idx],
        params.α[idx:idx],
        params.noise_std[idx:idx],
    )
end

function join_slices(ps::AbstractVector{<:IBNNParams{Nothing}})
    return IBNNParams(
        nothing,
        hcat(getfield.(ps, Ref(:λ))...),
        vcat(getfield.(ps, Ref(:α))...),
        vcat(getfield.(ps, Ref(:noise_std))...),
    )
end
function join_slices(ps::AbstractVector{<:IBNNParams{<:AbstractVector{<:Real}}})
    return IBNNParams(
        vcat(getfield.(ps, Ref(:ω))...),
        hcat(getfield.(ps, Ref(:λ))...),
        vcat(getfield.(ps, Ref(:α))...),
        vcat(getfield.(ps, Ref(:noise_std))...),
    )
end

struct IBNNPosterior <: ModelPosteriorSlice{IBNN}
    f::Function
end

function model_posterior_slice(model::IBNN, params::IBNNParams, data::ExperimentData, slice::Int)
    f = _model_posterior_slice(model, params, data, slice)
    return IBNNPosterior(f)
end

function _model_posterior_slice(model::IBNN, params::IBNNParams, data::ExperimentData, slice::Int)
    @warn "I-BNN posterior CANNOT be evaluated in parallel!"

    # slice data & params
    y = data.Y[slice,:]
    ω = isnothing(params.ω) ? nothing : params.ω[slice]
    λ = params.λ[:,slice]
    α = params.α[slice]
    noise_std = params.noise_std[slice]

    base_size = size(data.X, 2)
    augment_size = 1
    K_mem = _allocate_kernel_matrix(depth(model), base_size, augment_size)
    K_base = _calculate_kernel_matrix_base!(K_mem, model, data.X; ω, λ, α)

    inv_C = inv(K_base + (noise_std^2 * I))

    function post(x::AbstractVector{<:Real})
        _, K_base_aug, K_aug = _calculate_kernel_matrix_augment!(K_mem, model, data.X, @view x[:,:]; ω, λ, α)

        μ = K_base_aug' * inv_C * y
        Σ = K_aug - K_base_aug' * inv_C * K_base_aug

        μ_ = μ[1]
        σ_ = Σ[1,1] |> _clip_var |> sqrt
        return μ_, σ_ # ::Tuple{<:Real, <:Real}
    end
    function post(X::AbstractMatrix{<:Real})
        N = size(X, 2)
        if augment_size < N
            augment_size = N
            K_mem = _reallocate_kernel_matrix(K_mem, base_size, augment_size)
        end

        _, K_base_aug, K_aug = _calculate_kernel_matrix_augment!(K_mem, model, data.X, X; ω, λ, α)

        μ = K_base_aug' * inv_C * y
        Σ = K_aug - K_base_aug' * inv_C * K_base_aug

        # TODO check positive-definiteness of Σ ?
        return μ, Σ # ::Tuple{<:AbstractVector{<:Real}, <:AbstractMatrix{<:Real}}
    end
end

function _allocate_kernel_matrix(model_depth, base_size, augment_size=0)
    total_size = base_size + augment_size
    K = zeros((total_size, total_size, 1 + model_depth + 1))
    return K
end
function _reallocate_kernel_matrix(K, base_size, augment_size)
    total_size = base_size + augment_size
    K_ = zeros((total_size, total_size, size(K, 3)))
    K_[1:base_size, 1:base_size, :] = K[1:base_size, 1:base_size, :]
    return K_
end

function _calculate_kernel_matrix_base!(K, model, X;
    ω = nothing,
    λ = ones(size(X, 1)),
    α = 1.,
)
    x_dim = size(X, 1)
    base_size = size(X, 2)

    ω2 = isnothing(ω) ? nothing : ω^2

    # lengthscales
    X = X ./ λ

    @assert size(K, 3) == (1 + depth(model) + 1)
    @assert size(K, 1) == size(K, 2) >= base_size

    # K0
    for i in 1:base_size
        for j in 1:i
            K[i,j,1] = K[j,i,1] = _calc_K0(model, ω2, X[:,i], X[:,j], x_dim)
        end
    end

    # K1,...,KL
    for l in 1:depth(model)
        layer = model.layers[l]
        for i in 1:base_size
            for j in 1:i
                K[i,j,l+1] = K[j,i,l+1] = _calc_Kl(layer, ω2, K[i,i,l], K[i,j,l], K[j,j,l])
            end
        end
    end

    # K (with amplitude)
    # @assert size(K, 3) == (1 + depth(model) + 1)
    K[1:base_size, 1:base_size, depth(model)+2] .= K[1:base_size, 1:base_size, depth(model)+1]
    K[1:base_size, 1:base_size, depth(model)+2] .*= α^2

    # end == depth(model)+2
    K_base = @view K[1:base_size, 1:base_size, end]
    return K_base
end
function _calculate_kernel_matrix_augment!(K, model, X, X_;
    ω = nothing,
    λ = ones(size(X, 1)),
    α = 1.,
)
    x_dim = size(X, 1)
    base_size = size(X, 2)
    augment_size = size(X_, 2)
    total_size = base_size + augment_size
    
    ω2 = isnothing(ω) ? nothing : ω^2

    @assert size(K, 3) == (1 + depth(model) + 1)
    @assert size(K, 1) == size(K, 2) >= total_size

    X_all = hcat(X, X_)
    
    # lengthscales
    X_all ./= λ

    # K0
    for i in (base_size + 1):total_size
        for j in 1:i
            K[i,j,1] = K[j,i,1] = _calc_K0(model, ω2, X_all[:,i], X_all[:,j], x_dim)
        end
    end

    # K1,...,KL
    for l in 1:depth(model)
        layer = model.layers[l]
        for i in (base_size + 1):total_size
            for j in 1:i
                K[i,j,l+1] = K[j,i,l+1] = _calc_Kl(layer, ω2, K[i,i,l], K[i,j,l], K[j,j,l])
            end
        end
    end

    # K (with amplitude)
    # @assert size(K, 3) == (1 + depth(model) + 1)
    K[1:total_size, 1:total_size, depth(model)+2] .= K[1:total_size, 1:total_size, depth(model)+1]
    K[1:total_size, 1:total_size, depth(model)+2] .*= α^2

    # end == depth(model)+2
    K_base = @view K[1:base_size, 1:base_size, end]
    K_base_aug = @view K[1:base_size, (base_size+1):total_size, end]
    K_aug = @view K[(base_size+1):total_size, (base_size+1):total_size, end]
    return K_base, K_base_aug, K_aug
end

# Either `ω2_b`, `ω2_w` are defined in each layer or `omega_priors` are defined in `IBNNPriors` (and thus `ω2` is not nothing).
function _calc_K0(model::IBNN{Nothing}, ω2::Real, xi, xj, x_dim)
    return ω2 + ω2 * ((xi' * xj) / x_dim)
end
function _calc_K0(model::IBNN{Float64}, ω2::Nothing, xi, xj, x_dim)
    return model.ω2_b0 + model.ω2_w0 * ((xi' * xj) / x_dim)
end

# Either `ω2_b`, `ω2_w` are defined in each layer or `omega_priors` are defined in `IBNNPriors` (and thus `ω2` is not nothing).
function _calc_Kl(layer::IBNNLayer{<:Any, Nothing}, ω2::Real, kii, kij, kjj)
    return ω2 + ω2 * F(layer.act_func, kii, kij, kjj)
end
function _calc_Kl(layer::IBNNLayer{<:Any, Float64}, ω2::Nothing, kii, kij, kjj)
    return layer.ω2_b + layer.ω2_w * F(layer.act_func, kii, kij, kjj)
end

function mean_and_var(post::IBNNPosterior, x::AbstractVector{<:Real})
    μ, σ = post.f(x)
    return μ, σ^2 # ::Tuple{<:Real, <:Real}
end
function mean_and_var(post::IBNNPosterior, X::AbstractMatrix{<:Real})
    μ, Σ = post.f(X)
    return μ, diag(Σ) # ::Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}
end

function mean_and_cov(post::IBNNPosterior, X::AbstractMatrix{<:Real})
    μ, Σ = post.f(X)
    return μ, Σ # ::Tuple{<:AbstractVector{<:Real}, <:AbstractMatrix{<:Real}}
end

function mean(post::IBNNPosterior, x::AbstractVector{<:Real})
    μ, _ = post.f(x)
    return μ # ::Real
end
function mean(post::IBNNPosterior, X::AbstractMatrix{<:Real})
    μ, _ = post.f(X)
    return μ # ::AbstractVector{<:Real}
end

function var(post::IBNNPosterior, x::AbstractVector{<:Real})
    _, σ = post.f(x)
    return σ^2 # ::Real
end
function var(post::IBNNPosterior, X::AbstractMatrix{<:Real})
    _, Σ = post.f(X)
    return diag(Σ) # ::AbstractVector{<:Real}
end

function cov(post::IBNNPosterior, X::AbstractMatrix{<:Real})
    _, Σ = post.f(X)
    return Σ # ::AbstractMatrix{<:Real}
end

function model_loglike(model::IBNN, data::ExperimentData)
    function loglike(params::IBNNParams)
        ll_data = data_loglike(model, params, data)
        ll_params = params_loglike(model, params)
        return ll_data + ll_params
    end
end

function data_loglike(model::IBNN, params::IBNNParams, data::ExperimentData)
    return sum(data_loglike_slice.(
        Ref(model),
        Ref(data.X),
        eachrow(data.Y),
        params.ω,
        eachcol(params.λ),
        params.α,
        params.noise_std,
    ))
end

function data_loglike_slice(
    model::IBNN,
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real},
    ω::Union{Nothing, Real},
    λ::AbstractVector{<:Real},
    α::Real,
    noise_std::Real,
)
    base_size = size(X, 2)
    K_mem = _allocate_kernel_matrix(depth(model), base_size)
    K_base = _calculate_kernel_matrix_base!(K_mem, model, X; ω, λ, α)

    C = K_base + noise_std^2 * I
    return logpdf(MvNormal(zero(y), C), y)
end

function params_loglike(model::IBNN, params::IBNNParams)
    return params_loglike(model.priors, params)
end
function params_loglike(priors::IBNNPriors, params::IBNNParams)
    ll_ω = isnothing(priors.omega_priors) ? 0. : sum(logpdf.(priors.omega_priors, params.ω))
    ll_λ = sum(logpdf.(priors.lengthscale_priors, eachcol(params.λ)))
    ll_α = sum(logpdf.(priors.amplitude_priors, params.α))
    ll_noise = sum(logpdf.(priors.noise_std_priors, params.noise_std))
    return ll_ω + ll_λ + ll_α + ll_noise
end

_params_sampler(model::IBNN) = _params_sampler(model.priors)

function _params_sampler(priors::IBNNPriors)
    function sample(rng::AbstractRNG)
        ω = isnothing(priors.omega_priors) ? nothing : rand.(Ref(rng), priors.omega_priors)
        λ = hcat(rand.(Ref(rng), priors.lengthscale_priors)...)
        α = rand.(Ref(rng), priors.amplitude_priors)
        noise_std = rand.(Ref(rng), priors.noise_std_priors)

        return IBNNParams(ω, λ, α, noise_std)
    end
end

function vectorizer(model::IBNN)
    is_dirac, dirac_vals = create_dirac_mask(param_priors(model))

    function vectorize(params::IBNNParams{Nothing})
        ps = vcat(
            vec(params.λ),
            params.α,
            params.noise_std,
        )

        ps = filter_diracs(ps, is_dirac)
        return ps
    end
    function vectorize(params::IBNNParams{<:AbstractVector{<:Real}})
        ps = vcat(
            params.ω,
            vec(params.λ),
            params.α,
            params.noise_std,
        )

        ps = filter_diracs(ps, is_dirac)
        return ps
    end

    function devectorize(params::IBNNParams{Nothing}, ps::AbstractVector{<:Real})
        ps = insert_diracs(ps, is_dirac, dirac_vals)
        
        λ_len, α_len, n_len = param_lengths(params)
        λ_shape = size(params.λ)

        λ = reshape(ps[1:λ_len], λ_shape)
        α = ps[λ_len+1:λ_len+α_len]
        noise_std = ps[λ_len+α_len+1:end]

        return IBNNParams(nothing, λ, α, noise_std)
    end
    function devectorize(params::IBNNParams{<:AbstractVector{<:Real}}, ps::AbstractVector{<:Real})
        ps = insert_diracs(ps, is_dirac, dirac_vals)
        
        ω_len, λ_len, α_len, n_len = param_lengths(params)
        λ_shape = size(params.λ)

        ω = ps[1:ω_len]
        λ = reshape(ps[ω_len+1:ω_len+λ_len], λ_shape)
        α = ps[ω_len+λ_len+1:ω_len+λ_len+α_len]
        noise_std = ps[ω_len+λ_len+α_len+1:end]

        return IBNNParams(ω, λ, α, noise_std)
    end

    return vectorize, devectorize
end

# bijector(::IBNN) = InvSoftplus()
function bijector(model::IBNN)
    priors = param_priors(model)
    return default_bijector(priors)
end

function param_priors(model::IBNN)
    return param_priors(model.priors)
end
function param_priors(priors::IBNNPriors{Nothing})
    return vcat(
        priors.lengthscale_priors,
        priors.amplitude_priors,
        priors.noise_std_priors,
    )
end
function param_priors(priors::IBNNPriors{<:ThetaPriors})
    return vcat(
        priors.omega_priors,
        priors.lengthscale_priors,
        priors.amplitude_priors,
        priors.noise_std_priors,
    )
end
