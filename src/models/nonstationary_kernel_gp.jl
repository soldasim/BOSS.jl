
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
} <: ModelParams{NonstationaryKernelGP}
    θ::T
    σ::N
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
    )
end

function join_slices(slices::AbstractVector{<:NonstationaryKernelGPParams})
    θ = vcat(getfield.(slices, Ref(:θ))...)
    σ = vcat(getfield.(slices, Ref(:σ))...)

    return NonstationaryKernelGPParams(θ, σ)
end

function model_posterior_slice(model::NonstationaryKernelGP, params::NonstationaryKernelGPParams, data::ExperimentData, slice::Int)    
    gp = finite_nonkgp(model, params, data, slice)
    gp_post = AbstractGPs.posterior(gp, data.Y[slice,:])
    return GaussianProcessPosterior(gp_post) # -> gaussian_process.jl
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

function _params_sampler(model::NonstationaryKernelGP)
    function sample_params(rng::AbstractRNG)
        θ = rand.(Ref(rng), model.θ_priors)
        σ = rand.(Ref(rng), model.noise_std_priors)

        return NonstationaryKernelGPParams(θ, σ)
    end
end

function vectorizer(model::NonstationaryKernelGP)
    is_dirac, dirac_vals = create_dirac_mask(param_priors(model))

    function vectorize(params::NonstationaryKernelGPParams)
        ps = vcat(
            params.θ,
            params.σ,
        )

        ps = filter_diracs(ps, is_dirac)
        return ps
    end

    function devectorize(params::NonstationaryKernelGPParams, ps::AbstractVector{<:Real})
        θ_len = length(params.θ)
        ps = insert_diracs(ps, is_dirac, dirac_vals)

        θ = ps[1:θ_len]
        σ = ps[θ_len+1:end]

        return NonstationaryKernelGPParams(θ, σ)
    end

    return vectorize, devectorize
end

function bijector(model::NonstationaryKernelGP)
    _theta_bijs = (b -> b ? InvSoftplus() : identity).(model.softplus_params)
    noise_bij = default_bijector(model.noise_std_priors)

    if allequal(_theta_bijs)
        if noise_bij isa NoBijector
            return _theta_bijs[1]
        else
            return Stacked(
                [_theta_bijs[1], noise_bij],
                ranges([length(_theta_bijs), noise_bij.length_in]),
            )
        end
    else
        if noise_bij isa NoBijector
            return Stacked(
                _theta_bijs,
            )
        else
            return Stacked(
                vcat(_theta_bijs, noise_bij),
                ranges(vcat(fill(1, length(_theta_bijs)), noise_bij.length_in)),
            )
        end
    end
end

function param_priors(model::NonstationaryKernelGP)
    return vcat(
        model.θ_priors,
        model.noise_std_priors,
    )
end
