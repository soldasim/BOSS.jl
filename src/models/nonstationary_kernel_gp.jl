
"""
    NonstationaryKernelGP(; kwargs...)

TODO
"""
@kwdef struct NonstationaryKernelGP{
    M<:Union{Nothing, Function},
} <: SurrogateModel
    mean::M = nothing
    k::Function
    θ_priors::AbstractVector{UnivariateDistribution}
    softplus_params::AbstractVector{Bool} = fill(false, length(θ_priors))
    noise_std_priors::NoiseStdPriors

    function NonstationaryKernelGP(mean::M, k, θ_priors, softplus_params, noise_std_priors) where {M}
        @assert length(θ_priors) == length(softplus_params)
        return new{M}(mean, k, θ_priors, softplus_params, noise_std_priors)
    end
end

"""
    NonstationaryKernelGP(...)

TODO
"""
struct NonstationaryKernelGPParams{
    T<:AbstractVector{<:Real},
    N<:AbstractVector{<:Real},
    B<:AbstractVector{Bool},
} <: ModelParams{NonstationaryKernelGP}
    θ::T
    σ::N
    softplus_params::B
end

# TODO
# make_discrete

sliceable(::NonstationaryKernelGP) = true

function slice(model::NonstationaryKernelGP, idx::Int)
    y_dim_ = length(model.noise_std_priors)
    @assert length(model.θ_priors) % y_dim_ == 0
    θ_slice_len = Int(length(model.θ_priors) / y_dim_)
    idxs = (idx-1)*θ_slice_len+1:idx*θ_slice_len

    return NonstationaryKernelGP(
        mean_slice(model.mean, idx),
        model.k,
        model.θ_priors[idxs],
        model.softplus_params[idxs],
        model.noise_std_priors[idx:idx],
    )
end

function slice(params::NonstationaryKernelGPParams, idx::Int)
    y_dim_ = length(params.σ)
    @assert length(params.θ) % y_dim_ == 0
    θ_slice_len = Int(length(params.θ) / y_dim_)
    idxs = (idx-1)*θ_slice_len+1:idx*θ_slice_len

    return NonstationaryKernelGPParams(
        params.θ[idxs],
        params.σ[idx:idx],
        params.softplus_params[idxs],
    )
end

function join_slices(slices::AbstractVector{<:NonstationaryKernelGPParams})
    θ = vcat(getfield.(slices, Ref(:θ))...)
    σ = vcat(getfield.(slices, Ref(:σ))...)
    softplus_params = vcat(getfield.(slices, Ref(:softplus_params))...)

    return NonstationaryKernelGPParams(θ, σ, softplus_params)
end

function model_posterior_slice(model::NonstationaryKernelGP, params::NonstationaryKernelGPParams, data::ExperimentData, slice::Int)    
    gp = finite_nonkgp(model, params, data, slice)
    gp_post = AbstractGPs.posterior(gp, data.Y[slice,:])
    return model_posterior_slice(gp_post) # -> gaussian_process.jl
end

function finite_nonkgp(model::NonstationaryKernelGP, params::NonstationaryKernelGPParams, data::ExperimentData, slice::Int)
    mean_ = mean_slice(model.mean, slice) # -> gaussian_process.jl
    
    k = model.k
    θ = params.θ # already sliced, not whole θ
    kernel = CustomKernel((x, y) -> k(x, y, θ))

    σ = params.σ[slice]

    return finite_nonkgp(data.X, mean_, kernel, σ)
end

function finite_nonkgp(
    X::AbstractMatrix{<:Real},
    mean::Union{Nothing, Function},
    kernel::Kernel,
    noise_std::Real,
)
    return _GP(mean, kernel)(X, noise_std ^ 2; obsdim=2)
end

function data_loglike(model::NonstationaryKernelGP, data::ExperimentData)
    function ll_data(params::NonstationaryKernelGPParams)
        return sum(data_loglike_slice.(Ref(model), Ref(params), Ref(data), 1:y_dim(data)))
    end
end

function params_loglike(model::NonstationaryKernelGP)
    function ll_params(params::NonstationaryKernelGPParams)
        ll_θ = sum(logpdf.(model.θ_priors, params.θ))
        ll_σ = sum(logpdf.(model.noise_std_priors, params.σ))
        return ll_θ + ll_σ
    end
end

function data_loglike_slice(
    model::NonstationaryKernelGP,
    params::ModelParams,
    data::ExperimentData,
    slice::Int,
)
    gp = finite_nonkgp(model, params, data, slice)
    return logpdf(gp, data.Y[slice,:])
end

function sample_params(rng::AbstractRNG, model::NonstationaryKernelGP)
    θ = rand.(Ref(rng), model.θ_priors)
    σ = rand.(Ref(rng), model.noise_std_priors)

    return NonstationaryKernelGPParams(θ, σ, model.softplus_params)
end

function vectorize(params::NonstationaryKernelGPParams)
    return vcat(
        params.θ,
        params.σ,
    )
end

function devectorize(params::NonstationaryKernelGPParams, p::AbstractVector{<:Real})
    θ_len = length(params.θ)

    θ = p[1:θ_len]
    σ = p[θ_len+1:end]

    return NonstationaryKernelGPParams(θ, σ, params.softplus_params)
end

function bijector(params::NonstationaryKernelGPParams)
    return Stacked(vcat(
        (b -> b ? InvSoftplus() : identity).(params.softplus_params),
        fill(InvSoftplus(), length(params.σ)),
    ))
end

function param_priors(model::NonstationaryKernelGP, data::ExperimentData)
    return vcat(
        model.θ_priors,
        model.noise_std_priors,
    )
end
