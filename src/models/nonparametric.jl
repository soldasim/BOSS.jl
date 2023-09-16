using AbstractGPs
using Turing
using ForwardDiff
using Distributions

const MIN_PARAM_VALUE = 1e-6

# Workaround: https://discourse.julialang.org/t/zygote-gradient-does-not-work-with-abstractgps-custommean/87815/7
AbstractGPs.mean_vector(m::AbstractGPs.CustomMean, x::ColVecs) = map(m.f, eachcol(x.X))
AbstractGPs.mean_vector(m::AbstractGPs.CustomMean, x::RowVecs) = map(m.f, eachrow(x.X))
AbstractGPs.mean_vector(m::AbstractGPs.CustomMean, x::AbstractVector) = map(m.f, x)

"""
A kernel for dealing with discrete variables.
It is used as a wrapper around any other `AbstractGPs.Kernel`.

The field `dims` can be used to specify only some dimension as discrete.
All dimensions are considered as discrete if `dims == nothing`.

This function is used internally by the BOSS algorithm.
Use the `discrete` field of the `BOSS.OptimizationProblem` structure
to define discrete dimension instead of this structure.

See also: `BOSS.OptimizationProblem`(@ref)

# Examples:
```julia-repl
julia> DiscreteKernel(Matern52Kernel())
DiscreteKernel(Matern 5/2 Kernel (metric = Distances.Euclidean(0.0)), nothing)

julia> DiscreteKernel(Matern52Kernel(), [true, false, false])
DiscreteKernel(Matern 5/2 Kernel (metric = Distances.Euclidean(0.0)), Bool[1, 0, 0])
```
"""
struct DiscreteKernel{D} <: Kernel where {
    D<:Union{Missing, AbstractVector{Bool}}
}
    kernel::Kernel
    dims::D
end
DiscreteKernel(kernel::Kernel) = DiscreteKernel(kernel, missing)

function (dk::DiscreteKernel)(x1, x2)
    r(x) = discrete_round(dk.dims, x)
    dk.kernel(r(x1), r(x2))
end
(dk::DiscreteKernel)(x1::ForwardDiff.Dual, x2) = dk(x1.value, x2)
(dk::DiscreteKernel)(x1, x2::ForwardDiff.Dual) = dk(x1, x2.value)
(dk::DiscreteKernel)(x1::ForwardDiff.Dual, x2::ForwardDiff.Dual) = dk(x1.value, x2.value)

function KernelFunctions.with_lengthscale(dk::DiscreteKernel, lengthscale::Real)
    return DiscreteKernel(with_lengthscale(dk.kernel, lengthscale), dk.dims)
end
function KernelFunctions.with_lengthscale(dk::DiscreteKernel, lengthscales::AbstractVector{<:Real})
    return DiscreteKernel(with_lengthscale(dk.kernel, lengthscales), dk.dims)
end

"""
Construct a `BOSS.Nonparametric` model by adding the given mean function an existing nonparametric model.
"""
add_mean(m::Nonparametric{Nothing}, mean::Function) =
    Nonparametric(mean, m.kernel, m.length_scale_priors)

"""
Construct a new `BOSS.Nonparametric` model by wrapping its `kernel` in `BOSS.DiscreteKernel`
to define some dimensions as discrete.
"""
make_discrete(m::Nonparametric, discrete::AbstractVector{<:Bool}) =
    Nonparametric(m.mean, make_discrete(m.kernel, discrete), m.length_scale_priors)

make_discrete(k::Kernel, discrete::AbstractVector{<:Bool}) = DiscreteKernel(k, discrete)
make_discrete(k::DiscreteKernel, discrete::AbstractVector{<:Bool}) = make_discrete(k.kernel, discrete)

model_posterior(model::Nonparametric, data::ExperimentDataMLE) =
    model_posterior(model, data.X, data.Y, data.length_scales, data.noise_vars)

model_posterior(model::Nonparametric, data::ExperimentDataBI) =
    model_posterior.(Ref(model), Ref(data.X), Ref(data.Y), data.length_scales, eachcol(data.noise_vars))

"""
Return the posterior predictive distribution of the Gaussian Process.

The posterior is a function `mean, var = predict(x)`
which gives the mean and variance of the predictive distribution as a function of `x`.
"""
function model_posterior(
    model::Nonparametric,
    X::AbstractMatrix{NUM},
    Y::AbstractMatrix{NUM},
    length_scales::AbstractMatrix{NUM},
    noise_vars::AbstractVector{NUM},
) where {NUM<:Real}
    y_dim = length(noise_vars)
    means = isnothing(model.mean) ? fill(nothing, y_dim) : [x->model.mean(x)[i] for i in 1:y_dim]
    posts = model_posterior.(means, Ref(model.kernel), Ref(X), eachrow(Y), eachcol(length_scales), noise_vars)
    function posterior(x)
        ys = map(p->p(x), posts)
        first.(ys), last.(ys)
    end
end

function model_posterior(
    mean::Union{Nothing, Function},
    kernel::Kernel,
    X::AbstractMatrix{NUM},
    y::AbstractVector{NUM},
    length_scales::AbstractVector{NUM},
    noise_var::NUM,
) where {NUM<:Real}
    posterior_gp = AbstractGPs.posterior(finite_gp(mean, kernel, X, length_scales, noise_var), y)
    posterior(x) = first.(mean_and_var(posterior_gp(hcat(x))))
end

"""
Construct a finite GP via the AbstractGPs.jl library.
"""
function finite_gp(
    mean::Union{Nothing, Function},
    kernel::Kernel,
    X::AbstractMatrix{<:Real},
    length_scales::AbstractVector{<:Real},
    noise_var::Real;
    min_param_val::Real=MIN_PARAM_VALUE,
    min_noise::Real=MIN_PARAM_VALUE,
)
    # for numerical stability
    params = length_scales .+ min_param_val
    noise = noise_var + min_noise

    kernel = with_lengthscale(kernel, params)
    return finite_gp_(mean, kernel, X, noise)
end

finite_gp_(::Nothing, kernel::Kernel, X::AbstractMatrix{<:Real}, noise::Real) = GP(kernel)(X, noise)
finite_gp_(mean::Function, kernel::Kernel, X::AbstractMatrix{<:Real}, noise::Real) = GP(mean, kernel)(X, noise)

"""
Return the log-likelihood of the GP hyperparameters and the noise variance
as a function `ll = loglike(length_scales, noise_vars)`.
"""
function model_loglike(model::Nonparametric, noise_var_priors::AbstractVector{<:UnivariateDistribution}, data::ExperimentData)
    params_loglike = model_params_loglike(model, data.X, data.Y)
    noise_loglike(noise_vars) = mapreduce(p -> logpdf(p...), +, zip(noise_var_priors, noise_vars))
    loglike(length_scales, noise_vars) = params_loglike(length_scales, noise_vars) + noise_loglike(noise_vars)
end

"""
Return the log-likelihood of the GP hyperparameters (without the likelihood of the noise variance)
as a function `ll = loglike(length_scales, noise_vars)`.
"""
function model_params_loglike(model::Nonparametric, X::AbstractMatrix{NUM}, Y::AbstractMatrix{NUM}) where {NUM<:Real}
    y_dim = size(Y)[1]
    means = isnothing(model.mean) ? fill(nothing, y_dim) : [x->model.mean(x)[i] for i in 1:y_dim]

    function params_loglike(length_scales, noise_vars)
        function ll_data_dim(X, y, mean, length_scales, noise_var)
            gp = finite_gp(mean, model.kernel, X, length_scales, noise_var)
            logpdf(gp, y)
        end

        ll_data = mapreduce(p -> ll_data_dim(X, p...), +, zip(eachrow(Y), means, eachcol(length_scales), noise_vars))
        ll_length_scales = mapreduce(p -> logpdf(p...), +, zip(model.length_scale_priors, eachcol(length_scales)))
        ll_data + ll_length_scales
    end
end
