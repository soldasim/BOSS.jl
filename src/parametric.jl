using Turing

(model::Parametric)(θ::AbstractArray{<:Real}) = x -> model(x, θ)

function (model::LinModel)(x::AbstractArray{NUM}, θ::AbstractArray{NUM}) where {NUM<:Real}
    ϕs = model.lift(x)
    m = length(ϕs)

    ϕ_lens = length.(ϕs)
    θ_indices = vcat(0, partial_sums(ϕ_lens))
    
    y = [(θ[θ_indices[i]+1:θ_indices[i+1]])' * ϕs[i] for i in 1:m]
    return y
end

(m::NonlinModel)(x::AbstractArray{NUM}, θ::AbstractArray{NUM}) where {NUM<:Real} =
    m.predict(x, θ)

Base.convert(::Type{NonlinModel}, model::LinModel) =
    NonlinModel(
        (x, θ) -> model(x, θ),
        model.param_priors,
        model.param_count,
    )

model_posterior(model::Parametric, data::ExperimentDataMLE) =
    model_posterior(model, data.θ, data.noise_vars)

model_posterior(model::Parametric, data::ExperimentDataBI) = 
    model_posterior.(Ref(model), eachcol(data.θ), eachcol(data.noise_vars))

model_posterior(
    model::Parametric,
    θ::AbstractVector{NUM},
    noise_vars::AbstractVector{NUM}
) where {NUM<:Real} =
    predict(x) = model(x, θ), noise_vars

# Log-likelihood of model parameters and noise variance.
function model_loglike(model::Parametric, noise_var_priors::AbstractArray, data::ExperimentData)
    params_loglike = model_params_loglike(model, data.X, data.Y)
    noise_loglike(noise_vars) = mapreduce(p -> logpdf(p...), +, zip(noise_var_priors, noise_vars))
    loglike(θ, noise_vars) = params_loglike(θ, noise_vars) + noise_loglike(noise_vars)
end

# Log-likelihood of model parameters.
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
    return sums
end
