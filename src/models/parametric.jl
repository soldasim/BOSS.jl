using Turing
using Distributions

(model::Parametric)(θ::AbstractVector{<:Real}) = x -> model(x, θ)

function (model::LinModel)(x::AbstractVector{<:Real}, θ::AbstractVector{<:Real})
    x = discrete_round(model.discrete, x)

    ϕs = model.lift(x)
    m = length(ϕs)

    ϕ_lens = length.(ϕs)
    θ_indices = vcat(0, partial_sums(ϕ_lens))
    
    y = [(θ[θ_indices[i]+1:θ_indices[i+1]])' * ϕs[i] for i in 1:m]
    return y
end

function (m::NonlinModel)(x::AbstractVector{<:Real}, θ::AbstractVector{<:Real})
    x = discrete_round(m.discrete, x)
    return m.predict(x, θ)
end

Base.convert(::Type{NonlinModel}, model::LinModel) =
    NonlinModel(
        (x, θ) -> model(x, θ),
        model.param_priors,
        model.discrete,
    )

make_discrete(m::LinModel, discrete::AbstractVector{<:Bool}) =
    LinModel(m.lift, m.param_priors, discrete)
make_discrete(m::NonlinModel, discrete::AbstractVector{<:Bool}) =
    NonlinModel(m.predict, m.param_priors, discrete)

model_posterior(model::Parametric, data::ExperimentDataMLE) =
    model_posterior(model, data.θ, data.noise_vars)

model_posterior(model::Parametric, data::ExperimentDataBI) = 
    model_posterior.(Ref(model), eachcol(data.θ), eachcol(data.noise_vars))

"""
Return the posterior predictive distribution of the model.

The posterior is a function `mean, var = predict(x)`
which gives the mean and variance of the predictive distribution as a function of `x`.
"""
model_posterior(
    model::Parametric,
    θ::AbstractVector{NUM},
    noise_vars::AbstractVector{NUM}
) where {NUM<:Real} =
    predict(x) = model(x, θ), noise_vars

"""
Return the log-likelihood of the model parameters and the noise variance
as a function `ll = loglike(θ, noise_vars)`.
"""
function model_loglike(model::Parametric, noise_var_priors::AbstractVector{<:UnivariateDistribution}, data::ExperimentData)
    params_loglike = model_params_loglike(model, data.X, data.Y)
    noise_loglike(noise_vars) = mapreduce(p -> logpdf(p...), +, zip(noise_var_priors, noise_vars))
    loglike(θ, noise_vars) = params_loglike(θ, noise_vars) + noise_loglike(noise_vars)
end

"""
Return the log-likelihood of the model parameters (without the likelihood of the noise variance)
as a function `ll = loglike(θ, noise_vars)`.
"""
function model_params_loglike(model::Parametric, X::AbstractMatrix{NUM}, Y::AbstractMatrix{NUM}) where {NUM<:Real}
    function params_loglike(θ, noise_vars)
        ll_datum(x, y) = logpdf(MvNormal(model(x, θ), noise_vars), y)
        
        ll_data = mapreduce(d -> ll_datum(d...), +, zip(eachcol(X), eachcol(Y)))
        ll_params = mapreduce(p -> logpdf(p...), +, zip(model.param_priors, θ))
        ll_data + ll_params
    end
end

function partial_sums(array::AbstractArray)
    isempty(array) && return empty(array)
    s = zero(first(array))
    sums = [(s += val) for val in array]
end
