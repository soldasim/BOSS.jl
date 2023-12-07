
make_discrete(m::Semiparametric, discrete::AbstractVector{<:Bool}) =
    Semiparametric(make_discrete(m.parametric, discrete), make_discrete(m.nonparametric, discrete))

model_posterior(model::Semiparametric, data::ExperimentDataMLE) =
    model_posterior(model, data.X, data.Y, data.θ, data.length_scales, data.noise_vars)

model_posterior(model::Semiparametric, data::ExperimentDataBI) =
    model_posterior.(Ref(model), Ref(data.X), Ref(data.Y), eachcol(data.θ), data.length_scales, eachcol(data.noise_vars))

function model_posterior(
    model::Semiparametric,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    θ::AbstractVector{<:Real},
    length_scales::AbstractMatrix{<:Real},
    noise_vars::AbstractVector{<:Real},   
)
    return model_posterior(add_mean(model.nonparametric, model.parametric(θ)), X, Y, length_scales, noise_vars)
end

function model_loglike(model::Semiparametric, noise_var_priors, data::ExperimentData)
    function loglike(θ, length_scales, noise_vars)
        ll_params = model_params_loglike(model, θ, length_scales)
        ll_data = model_data_loglike(model, θ, length_scales, noise_vars, data.X, data.Y)
        ll_noise = noise_loglike(noise_var_priors, noise_vars)
        return ll_params + ll_data + ll_noise
    end
end

function model_params_loglike(model::Semiparametric, θ::AbstractVector{<:Real}, length_scales::AbstractMatrix{<:Real})
    ll_param = model_params_loglike(model.parametric, θ)
    ll_gp = model_params_loglike(model.nonparametric, length_scales)
    return ll_param + ll_gp
end

function model_data_loglike(
    model::Semiparametric,
    θ::AbstractVector{<:Real},
    length_scales::AbstractMatrix{<:Real},
    noise_vars::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    return model_data_loglike(
        add_mean(model.nonparametric, model.parametric(θ)),
        length_scales,
        noise_vars,
        X,
        Y,
    )
end
