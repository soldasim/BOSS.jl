
"""
    ParametrizedGP(; kwargs...)

A Gaussian Process model parametrized by the data with a univariate output.

`ParametrizedGP` is *not* a subtype of `SurrogateModel`. It is a special model
used to handle parameters of the `NonstationaryGP` model.

The outputs `Y` are considered parameters of the model. This makes `ParametrizedGP`
a parametric model, not a nonparametric model as a standard GP would be.
It can be considered a complex prior for the hyperparameters of `NonstationaryGP`.

The mean and the amplitude of the GP are fixed to `0` and `1` respectively.
The prior on the modeled variables is defined via the `target_dist` and `act_func`
together with the `lengthscale_priors`.

The prediction for the modeled variable is calculated as follows;
- The outputs `y` are modeled by the GP posterior
    (mean `0`, amplitude `1`, lengthscaled given by the `lengthscale_priors`).
- The outputs `y` are transformed into new variables `z` so that each `z_i`
    is distributed according the `target_dist`.
- Finally, each `z_i` is mapped via the `act_func`.

The `noise_std` hyperparameter should be set to small values. Technically, it should
be zero, but small positive values are required for numerical stability.

# Keywords
- `mean::Union{Nothing, Real, Function}`: Used as the mean function for the GP.
        Defaults to `nothing` equivalent to `0` or `x -> 0`.
- `kernel::Kernel`: The kernel used in the GP. Defaults to the `Matern32Kernel()`.
- `lengthscale_priors::MultivariateDistribution`: The prior distribution
        for the length scales of the GP.
- `target_dist::UnivariateDistribution`: The distribution of the modeled variable.
- `act_func::Function`: The activation function applied on the output
        of the model. Defaults to `identity`.
- `noise_std::Real`: The noise standard deviation of the GP.
"""
@kwdef struct ParametrizedGP{
    D<:Union{Nothing, UnivariateDistribution},
}
    kernel::Kernel = Matern32Kernel()
    lengthscale_prior::MultivariateDistribution
    target_dist::D = nothing
    act_func::Function = identity
    noise_std::Real = 0.
end

"""
    ParametrizedGPParams(X, μ, L, yϵ, λ)

The parameters of the [`ParametrizedGP`](@ref) model.

# Parameters
- `X::AbstractMatrix{<:Real}`: The inputs of the GP.
        (Not parameters of the model, but need to be saved as context for `y`.)
- `μ::AbstractVector{<:Real}`: The mean of the outputs `y`.
        Used together with `L` to whiten the outputs `y` into the indepedent `yϵ`.
- `L::AbstractMatrix{<:Real}`: The Cholesky decomposition of the covariance matrix of the outputs `y`.
        Used to whiten the outputs `y` into the indepedent `yϵ`.
- `yϵ::AbstractVector{<:Real}`: The outputs of the GP, whitened to be independent.
        (Here, considered parameters of the model.)
- `λ::AbstractVector{<:Real}`: The length scales of the GP.
"""
struct ParametrizedGPParams{
    TX<:AbstractMatrix{<:Real},
    TM<:AbstractVector{<:Real},
    TL<:AbstractMatrix{<:Real},
    Ty<:AbstractVector{<:Real},
    L<:AbstractVector{<:Real},
}
    X::TX
    μ::TM
    L::TL
    yϵ::Ty
    λ::L
end

# Base.length(params::ParametrizedGPParams) = length(params.yϵ) + length(params.λ)

# function make_discrete(model::ParametrizedGP, discrete::AbstractVector{<:Bool})
#     return ParametrizedGP(
#         model.mean,
#         make_discrete(model.kernel, discrete),
#         model.lengthscale_prior,
#         model.amplitude_prior,
#         model.noise_std,
#     )
# end

function model_posterior(model::ParametrizedGP, params::ParametrizedGPParams, data::ExperimentData)
    # de-whiten the gp outputs
    y = params.L * params.yϵ + params.μ
    
    gp_post = AbstractGPs.posterior(
        finite_param_gp(model, params),
        y,
    )
    ft = construct_variable_transform(model.target_dist)
    a = model.act_func

    function post(x::AbstractVector{<:Real})
        μ = mean(gp_post(hcat(x); obsdim=2))[1]
        return μ |> ft |> a
    end
end

# cheap posterior which can be evaluated at the points in the dataset only
function model_posterior_lookup(model::ParametrizedGP, params::ParametrizedGPParams, data::ExperimentData)
    # de-whiten the outputs
    y = params.L * params.yϵ + params.μ

    ft = construct_variable_transform(model.target_dist)
    a = model.act_func

    vals = y .|> ft .|> a
    lookup = Dict(Pair.(eachcol(params.X), vals))

    post(x::AbstractVector{<:Real}) = lookup[x]
end

function construct_variable_transform(::Nothing)
    return identity
end
function construct_variable_transform(target_dist::UnivariateDistribution)
    py = Normal(0, 1)
    pz = target_dist

    function ft(y::Real)
        u = cdf(py, y)
        z = quantile(pz, u)
        return z
    end
end

function finite_param_gp(model::ParametrizedGP, params::ParametrizedGPParams)
    return finite_param_gp(
        params.X,
        model.kernel,
        params.λ,
        model.noise_std,
    )
end
function finite_param_gp(
    X::AbstractMatrix{<:Real},
    kernel::Kernel,
    λ::AbstractVector{<:Real},
    noise_std::Real,
)
    return finite_gp(
        X,
        0, # mean
        kernel,
        λ,
        1, # amplitude
        noise_std,
    ) # -> gaussian_process.jl
end

function params_loglike(model::ParametrizedGP, data::ExperimentData)
    if (model.lengthscale_prior isa Product{<:Any, <:Dirac})
        # all hyperparameters of the `ParametrizedGP` are fixed
        # the kernel matrix can be precomputed
        
        # λ = getfield.(model.lengthscale_prior.v, Ref(:value))

        # gp = finite_param_gp(data.X, model.kernel, λ, model.noise_std)
        # μ, Σ = mean_and_cov(gp)
        # L = cholesky(Σ).L

        # only `y` is being fitted
        function loglike_y(params::ParametrizedGPParams)
            ll_y = logpdf(MvNormal(zero(params.μ), I(length(params.μ))), params.yϵ)
            return ll_y
        end
    
    else
        # some hyperparameters of the `ParametrizedGP` are being fitted
        # the kernel matrix must be re-computed for each evaluation
        function loglike_full(params::ParametrizedGPParams)
            @assert false # L cannot be precomputed wihout fixed hyperparameters
            
            # gp = finite_param_gp(model, params)

            # ll_y = logpdf(gp, params.y)
            # ll_λ = logpdf(model.lengthscale_prior, params.λ)
            # ll_α = logpdf(model.amplitude_prior, params.α)
            # return ll_y + ll_λ + ll_α
        end
    end
end

function _params_sampler(model::ParametrizedGP, data::ExperimentData)
    # must have fixed hyperparameter to be able to precompute L for output whitening
    @assert model.lengthscale_prior isa Product{<:Any, <:Dirac}
    
    # diracs
    λ = rand(model.lengthscale_prior)

    gp = finite_param_gp(data.X, model.kernel, λ, model.noise_std)
    μ, K = AbstractGPs.mean_and_cov(gp)
    L = cholesky(K).L

    len = length(μ)
    p_yϵ = MvNormal(zeros(len), I(len))

    return function sample(rng)
        yϵ = rand(rng, p_yϵ)

        return ParametrizedGPParams(
            data.X,
            μ,
            L,
            yϵ,
            λ,
        )
    end
end

function vectorizer(model::ParametrizedGP, data::ExperimentData)
    priors = vcat(
        fill(nothing, length(data)),
        model.lengthscale_prior,
    )
    is_dirac, dirac_vals = create_dirac_mask(priors)

    params = _params_sampler(model, data)(Random.default_rng())
    y_len = length(params.yϵ)
    λ_len = length(params.λ)
    y_ran, λ_ran = ranges([y_len, λ_len])

    function vectorize(params::ParametrizedGPParams)
        ps = vcat(
            params.yϵ,
            params.λ,
        )
        ps = filter_diracs(ps, is_dirac)
        return ps
    end
    
    function devectorize(params::ParametrizedGPParams, ps::AbstractVector{<:Real})
        ps = insert_diracs(ps, is_dirac, dirac_vals)
        
        yϵ = @view ps[y_ran]
        λ = @view ps[λ_ran]
    
        return ParametrizedGPParams(params.X, params.μ, params.L, yϵ, λ)
    end

    return vectorize, devectorize
end

function bijector(model::ParametrizedGP, data::ExperimentData)
    yϵ_bij = identity
    
    hyper_priors = filter_dirac_priors([model.lengthscale_prior])
    hyper_bij = default_bijector(hyper_priors)

    if isnothing(hyper_bij)
        return Stacked(
            [yϵ_bij],
            [1:length(data)],
        )
    else
        return Stacked(
            [yϵ_bij, hyper_bij],
            ranges([length(data), hyper_bij.length_in])
        )
    end
end
