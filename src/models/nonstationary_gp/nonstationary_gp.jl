
"""
    NonstationaryGP(; kwargs...)

A Gaussian Process model with an option to model the length scales, amplitudes,
and/or noise standard deviations with additional GPs.

# Keywords
- `mean::Union{Nothing, AbstractVector{<:Real}, Function}`: Used as the mean function for the GP.
        Defaults to `nothing` equivalent to `x -> zeros(y_dim)`.
- `lengthscale_model::Union{LengthscalePriors, AbstractMatrix{<:ParametrizedGP}}`:
        The model used for the length scales of the GP.
        Define it as `LengthscalePriors` to use standard stationary lengthscales,
        or define it as a matrix of `ParametrizedGP` to use nonstationary lengthscales.
- `amplitude_model::Union{AmplitudePriors, AbstractVector{<:ParametrizedGP}}`:
        The model used for the amplitude hyperparameters of the GP.
        Define it as `AmplitudePriors` to use standard stationary amplitudes,
        or define it as a vector of `ParametrizedGP` to use nonstationary amplitudes.
- `noise_std_model::Union{NoiseStdPriors, AbstractVector{<:ParametrizedGP}}`:
        The model used for the noise standard deviations of the GP.
        Define it as `NoiseStdPriors` to use standard stationary noise stds,
        or define it as a vector of `ParametrizedGP` to use nonstationary noise stds.
"""
@kwdef struct NonstationaryGP{
    M<:Union{Nothing, AbstractVector{<:Real}, Function},
} <: SurrogateModel
    mean::M = nothing
    lengthscale_model::Union{LengthscalePriors, AbstractMatrix{<:ParametrizedGP}}
    amplitude_model::Union{AmplitudePriors, AbstractVector{<:ParametrizedGP}}
    noise_std_model::Union{NoiseStdPriors, AbstractVector{<:ParametrizedGP}}
    discrete::Union{Nothing, AbstractVector{<:Bool}} = nothing
end

"""
    NonstationaryGPParams(λ, α, σ)

The parameters of the [`NonstationaryGP`](@ref) model.

# Parameters
- `λ::AbstractMatrix{<:Union{Real, ParametrizedGPParams}}`:
        The length scales of the GP, or the parameters of the `ParametrizedGP`s
        used to model nonstationary lengthscales.
- `α::AbstractVector{<:Union{Real, ParametrizedGPParams}}`:
        The amplitudes of the GP, or the parameters of the `ParametrizedGP`s
        used to model nonstationary amplitudes.
- `σ::AbstractVector{<:Union{Real, ParametrizedGPParams}}`:
        The noise standard deviations of the GP, or the parameters of the `ParametrizedGP`s
        used to model nonstationary noise stds.
"""
struct NonstationaryGPParams{
    L<:AbstractMatrix{<:Union{Real, ParametrizedGPParams}},
    A<:AbstractVector{<:Union{Real, ParametrizedGPParams}},
    N<:AbstractVector{<:Union{Real, ParametrizedGPParams}},
} <: ModelParams{NonstationaryGP}
    λ::L
    α::A
    σ::N
end

struct NonstationaryKernel <: Kernel
    f_λ::Function
    f_α::Function
end

function (k::NonstationaryKernel)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    return nonstat_kernel(x, y, k.f_λ, k.f_α)
end

function nonstat_kernel(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    f_λ::Function,
    f_α::Function,
)
    return nonstat_kernel(x, y, f_λ(x), f_λ(y), f_α(x), f_α(y))
end

function nonstat_kernel(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    λx::AbstractVector{<:Real},
    λy::AbstractVector{<:Real},
    αx::Real,
    αy::Real,
)
    # TODO amplitudes ??
    return ((αx + αy) / 2)^2 * gibbs_kernel(x, y, λx, λy)
end

function gibbs_kernel(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    λx::AbstractVector{<:Real},
    λy::AbstractVector{<:Real},
)
    # mapreduce is slow
    # return mapreduce(gibbs_kernel, *, x, y, λx, λy)

    v = 1.
    for i in eachindex(x)
        v *= @inbounds gibbs_kernel(x[i], y[i], λx[i], λy[i])
    end
    return v
end
function gibbs_kernel(x::Real, y::Real, λx::Real, λy::Real)
    return sqrt((2 * λx * λy) / (λx^2 + λy^2)) * exp((-1) * norm(x - y)^2 / (λx^2 + λy^2))
end

function make_discrete(model::NonstationaryGP, discrete::AbstractVector{<:Bool})
    return NonstationaryGP(
        model.mean,
        make_discrete(model.lengthscale_model, discrete),
        make_discrete(model.amplitude_model, discrete),
        make_discrete(model.noise_std_model, discrete),
        discrete,
    )
end

make_discrete(m::AbstractVector{<:Distribution}, discrete::AbstractVector{<:Bool}) = m

sliceable(::NonstationaryGP) = true

function slice(model::NonstationaryGP, idx::Int)
    return NonstationaryGP(
        mean_slice(model.mean, idx),
        slice_param_model(model.lengthscale_model, idx),
        slice_param_model(model.amplitude_model, idx),
        slice_param_model(model.noise_std_model, idx),
        model.discrete,
    )
end

slice_param_model(m::AbstractVector{<:Distribution}, idx::Int) = m[idx:idx]
slice_param_model(m::AbstractVector{<:ParametrizedGP}, idx::Int) = m[idx:idx]
slice_param_model(m::AbstractMatrix{<:ParametrizedGP}, idx::Int) = m[:,idx:idx]

function slice(params::NonstationaryGPParams, idx::Int)
    return NonstationaryGPParams(
        params.λ[:,idx:idx],
        params.α[idx:idx],
        params.σ[idx:idx],
    )
end

function join_slices(slices::AbstractVector{<:NonstationaryGPParams})
    λ = hcat(getfield.(slices, Ref(:λ))...)
    α = vcat(getfield.(slices, Ref(:α))...)
    σ = vcat(getfield.(slices, Ref(:σ))...)

    return NonstationaryGPParams(λ, α, σ)
end

function model_posterior_slice(model::NonstationaryGP, params::NonstationaryGPParams, data::ExperimentData, slice::Int)    
    gp = finite_nongp(model, params, data, slice)
    gp_post = AbstractGPs.posterior(gp, data.Y[slice,:])
    return GaussianProcessPosterior(gp_post) # -> gaussian_process.jl
end

function finite_nongp(model::NonstationaryGP, params::NonstationaryGPParams, data::ExperimentData, slice::Int)
    f_λ = _param_posterior_slice(model.lengthscale_model, params.λ, data, slice)
    f_α = _param_posterior_slice(model.amplitude_model, params.α, data, slice)
    f_σ = _param_posterior_slice(model.noise_std_model, params.σ, data, slice)

    mean_ = mean_getindex(model.mean, slice) # -> gaussian_process.jl

    return finite_nongp(data.X, mean_, f_λ, f_α, f_σ, model.discrete)
end

# Can only be evaluated at the points present in the dataset,
# but is significantly cheaper to compute as it does no require
# constructing the kernel matrices of the underlying parametric GPs.
function finite_nongp_lookup(model::NonstationaryGP, params::NonstationaryGPParams, data::ExperimentData, slice::Int)
    f_λ = _param_posterior_slice_lookup(model.lengthscale_model, params.λ, data, slice)
    f_α = _param_posterior_slice_lookup(model.amplitude_model, params.α, data, slice)
    f_σ = _param_posterior_slice_lookup(model.noise_std_model, params.σ, data, slice)

    mean_ = mean_getindex(model.mean, slice) # -> gaussian_process.jl

    return finite_nongp(data.X, mean_, f_λ, f_α, f_σ, model.discrete)
end

function finite_nongp(
    X::AbstractMatrix{<:Real},
    mean::Union{Nothing, Real, Function},
    f_λ::Function,
    f_α::Function,
    f_σ::Function,
    discrete::Union{Nothing, AbstractVector{<:Bool}}
)
    kernel = NonstationaryKernel(f_λ, f_α)
    kernel = make_discrete(kernel, discrete)
    noise_std = f_σ.(eachcol(X))

    return _GP(mean, kernel)(X, noise_std .^ 2; obsdim=2)
end

_GP(mean::Nothing, kernel::Kernel) = GP(kernel)
_GP(mean, kernel::Kernel) = GP(mean, kernel)

function _param_posterior_slice(model::AbstractVector{<:UnivariateDistribution}, params::AbstractVector{<:Real}, data::ExperimentData, slice::Int)
    p = params[slice]
    return (x::AbstractVector{<:Real}) -> p
end
function _param_posterior_slice(model::AbstractVector{<:MultivariateDistribution}, params::AbstractMatrix{<:Real}, data::ExperimentData, slice::Int)
    p = params[:,slice]
    return (x::AbstractVector{<:Real}) -> p
end
function _param_posterior_slice(model::AbstractVector{<:ParametrizedGP}, params::AbstractVector{<:ParametrizedGPParams}, data::ExperimentData, slice::Int)
    return model_posterior(model[slice], params[slice], data) # -> parameterized_gp.jl
end
function _param_posterior_slice(model::AbstractMatrix{<:ParametrizedGP}, params::AbstractMatrix{<:ParametrizedGPParams}, data::ExperimentData, slice::Int)
    posts = model_posterior.(model[:,slice], params[:,slice], Ref(data)) # -> parameterized_gp.jl
    return (x::AbstractVector{<:Real}) -> apply.(posts, Ref(x))
end

# same as `_param_posterior_slice`
function _param_posterior_slice_lookup(model::AbstractVector{<:UnivariateDistribution}, params::AbstractVector{<:Real}, data::ExperimentData, slice::Int)
    p = params[slice]
    return (x::AbstractVector{<:Real}) -> p
end
function _param_posterior_slice_lookup(model::AbstractVector{<:MultivariateDistribution}, params::AbstractMatrix{<:Real}, data::ExperimentData, slice::Int)
    p = params[:,slice]
    return (x::AbstractVector{<:Real}) -> p
end
# cheaper than `_param_posterior_slice`
function _param_posterior_slice_lookup(model::AbstractVector{<:ParametrizedGP}, params::AbstractVector{<:ParametrizedGPParams}, data::ExperimentData, slice::Int)
    return model_posterior_lookup(model[slice], params[slice], data) # -> parameterized_gp.jl
end
function _param_posterior_slice_lookup(model::AbstractMatrix{<:ParametrizedGP}, params::AbstractMatrix{<:ParametrizedGPParams}, data::ExperimentData, slice::Int)
    posts = model_posterior_lookup.(model[:,slice], params[:,slice], Ref(data)) # -> parameterized_gp.jl
    return (x::AbstractVector{<:Real}) -> apply.(posts, Ref(x))
end

function data_loglike(model::NonstationaryGP, data::ExperimentData)
    function ll_data(params::NonstationaryGPParams)
        return sum(data_loglike_slice.(Ref(model), Ref(params), Ref(data), 1:y_dim(data)))
    end
end

function data_loglike_slice(
    model::NonstationaryGP,
    params::NonstationaryGPParams,
    data::ExperimentData,
    slice::Int,
)
    gp = finite_nongp_lookup(model, params, data, slice)
    return logpdf(gp, data.Y[slice,:])
end

function params_loglike(model::NonstationaryGP, data::ExperimentData)
    loglike_λ = _params_loglike(model.lengthscale_model, data)
    loglike_α = _params_loglike(model.amplitude_model, data)
    loglike_σ = _params_loglike(model.noise_std_model, data)

    function ll_params(params::NonstationaryGPParams)
        ll_λ = loglike_λ(params.λ)
        ll_α = loglike_α(params.α)
        ll_σ = loglike_σ(params.σ)
        return ll_λ + ll_α + ll_σ
    end
end

function _params_loglike(m::AbstractArray{<:ParametrizedGP}, data::ExperimentData)
    loglikes = params_loglike.(m, Ref(data)) # -> parameterized_gp.jl

    function loglike(params::AbstractArray{<:ParametrizedGPParams})
        return sum(apply.(loglikes, params))
    end
end
function _params_loglike(m::AbstractVector{<:UnivariateDistribution}, data::ExperimentData)
    function loglike(params::AbstractVector{<:Real})
        return sum(logpdf.(m, params))
    end
end
function _params_loglike(m::AbstractVector{<:MultivariateDistribution}, data::ExperimentData)
    function loglike(params::AbstractMatrix{<:Real})
        return sum(logpdf.(m, eachcol(params)))
    end
end

function _params_sampler(model::NonstationaryGP, data::ExperimentData)
    λ_sampler = _params_sampler(model.lengthscale_model, data)
    α_sampler = _params_sampler(model.amplitude_model, data)
    σ_sampler = _params_sampler(model.noise_std_model, data)

    function sample(rng::AbstractRNG)
        λ_params = λ_sampler(rng)
        α_params = α_sampler(rng)
        σ_params = σ_sampler(rng)

        return NonstationaryGPParams(λ_params, α_params, σ_params)
    end
end

function _params_sampler(m::AbstractArray{<:ParametrizedGP}, data::ExperimentData)
    samplers = _params_sampler.(m, Ref(data)) # -> parameterized_gp.jl
    sample(rng::AbstractRNG) = apply.(samplers, Ref(rng))
end
_params_sampler(m::AbstractVector{<:UnivariateDistribution}, data::ExperimentData) = (rng::AbstractRNG) -> rand.(Ref(rng), m)
_params_sampler(m::AbstractVector{<:MultivariateDistribution}, data::ExperimentData) = (rng::AbstractRNG) -> hcat(rand.(Ref(rng), m)...)

function vectorizer(model::NonstationaryGP, data::ExperimentData)
    λ_vec, λ_devec = _vectorizer(model.lengthscale_model, data)
    α_vec, α_devec = _vectorizer(model.amplitude_model, data)
    σ_vec, σ_devec = _vectorizer(model.noise_std_model, data)

    params = params_sampler(model, data)()
    λ_len = length(λ_vec(params.λ))
    α_len = length(α_vec(params.α))
    σ_len = length(σ_vec(params.σ))
    λ_ran, α_ran, σ_ran = ranges([λ_len, α_len, σ_len])

    function vectorize(params::NonstationaryGPParams)
        λ_p = λ_vec(params.λ)
        α_p = α_vec(params.α)
        σ_p = σ_vec(params.σ)

        return vcat(λ_p, α_p, σ_p)
    end
    
    function devectorize(params::NonstationaryGPParams, p::AbstractVector{<:Real})
        λ = λ_devec(params.λ, @view p[λ_ran])
        α = α_devec(params.α, @view p[α_ran])
        σ = σ_devec(params.σ, @view p[σ_ran])
    
        return NonstationaryGPParams(λ, α, σ)
    end
    
    return vectorize, devectorize
end

function _vectorizer(m::AbstractArray{<:ParametrizedGP}, data::ExperimentData)
    ret = vectorizer.(m, Ref(data)) # -> parameterized_gp.jl
    vecs = first.(ret)
    devecs = second.(ret)
    
    params = apply.(_params_sampler.(m, Ref(data)), Ref(Random.default_rng()))
    lens = length.(apply.(vecs, params)) |> vec
    rans = ranges(lens)

    function vectorize(params::AbstractArray{<:ParametrizedGPParams})
        return vcat(apply.(vecs, params)...)
    end

    function devectorize(params::AbstractArray{<:ParametrizedGPParams}, ps::AbstractVector{<:Real})
        params_ = similar(params, ParametrizedGPParams)
        for i in 1:length(params)
            range_ = rans[i]
            params_[i] = devecs[i](params[i], @view ps[range_]) # -> parameterized_gp.jl
        end
        return params_
    end

    return vectorize, devectorize
end
function _vectorizer(m::AbstractVector{<:UnivariateDistribution}, data::ExperimentData)
    is_dirac, dirac_vals = create_dirac_mask(m)

    function vectorize(ps::AbstractVector{<:Real})
        ps = filter_diracs(ps, is_dirac)
        return ps
    end

    function devectorize(params::AbstractVector{<:Real}, ps::AbstractVector{<:Real})
        ps = insert_diracs(ps, is_dirac, dirac_vals)
        return ps
    end

    return vectorize, devectorize
end
function _vectorizer(m::AbstractVector{<:MultivariateDistribution}, data::ExperimentData)
    is_dirac, dirac_vals = create_dirac_mask(m)

    function vectorize(params::AbstractMatrix{<:Real})
        ps = vec(params)
        ps = filter_diracs(ps, is_dirac)
        return ps
    end

    function devectorize(params::AbstractMatrix{<:Real}, ps::AbstractVector{<:Real})
        ps = insert_diracs(ps, is_dirac, dirac_vals)
        params_ = reshape(ps, size(params))
        return params_
    end

    return vectorize, devectorize
end

function bijector(model::NonstationaryGP, data::ExperimentData)
    λ_bij = _bijector(model.lengthscale_model, data)
    α_bij = _bijector(model.amplitude_model, data)
    σ_bij = _bijector(model.noise_std_model, data)

    bijs = [filter(b -> !(b isa NoBijector), [λ_bij, α_bij, σ_bij])...]
    rans = ranges(getfield.(bijs, Ref(:length_in)))

    isempty(bijs) && return identity

    b = Stacked(
        bijs,
        rans,
    )
    b = simplify(b)
    return b
end

function _bijector(model::AbstractArray{<:ParametrizedGP}, data::ExperimentData)
    model_ = vec(model)
    bijs = bijector.(model_, Ref(data)) # -> parameterized_gp.jl
    rans = ranges(getfield.(bijs, Ref(:length_in)))
    return Stacked(bijs, rans)
end
function _bijector(model::AbstractVector{<:Distribution}, data::ExperimentData)
    return default_bijector(model)
end


# --- Reasonable Default Priors ---

# y ∼ N(0, 1) (+ correlations given by `lengthscale_prior`)
# z ∼ `target_dist`
# λ = `act_func(z)`
function default_lengthscale_model(bounds::AbstractBounds, y_dim::Int)
    @assert length(bounds[1]) == length(bounds[2])
    x_dim = length(bounds[1])
    lbs, ubs = bounds

    lengthscale_prior = Product(Dirac.((ubs .- lbs) ./ 3))

    function pgp(lb, ub)
        d = (ub - lb)
        
        λ_lb = d / 20
        λ_ub = d
        λ_d = (λ_ub - λ_lb)

        return ParametrizedGP(;
            kernel = GaussianKernel(),
            lengthscale_prior,
            # target_dist = Beta(2, 3), # map the GP prior from Normal(0, 1) to Beta(2, 3)
            target_dist = truncated(LogNormal(-1, 1); upper=1.), # map the GP prior from Normal(0, 1) to LogNormal(-1, 1)
            act_func = x -> λ_d * x + λ_lb, # map the lengthscale from [0, 1] to [λ_lb, λ_ub]
            noise_std = 1e-4,
        )
    end

    λ_priors = [pgp(lbs[x], ubs[x]) for x in 1:x_dim, y in 1:y_dim]
    return λ_priors
end
