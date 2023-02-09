using AbstractGPs
using Turing
using ForwardDiff

const MIN_PARAM_VALUE = 1e-6

# A workaround: https://discourse.julialang.org/t/zygote-gradient-does-not-work-with-abstractgps-custommean/87815/7
struct CustomMean{Tf} <: AbstractGPs.MeanFunction
    f::Tf
end

AbstractGPs._map_meanfunction(m::CustomMean, x::ColVecs) = map(m.f, eachcol(x.X))
AbstractGPs._map_meanfunction(m::CustomMean, x::RowVecs) = map(m.f, eachrow(x.X))
AbstractGPs._map_meanfunction(m::CustomMean, x::AbstractVector) = map(m.f, x)

"""
A kernel for dealing with discrete variables.
It is used as a wrapper around any other `AbstractGPs.Kernel`.

The field `dims` can be used to specify only some dimension as discrete.
All dimensions are considered as discrete if `dims == nothing`.

## Examples:
```julia-repl
julia> DiscreteKernel(Matern52Kernel())
DiscreteKernel(Matern 5/2 Kernel (metric = Distances.Euclidean(0.0)), nothing)

julia> DiscreteKernel(Matern52Kernel(), [true, false, false])
DiscreteKernel(Matern 5/2 Kernel (metric = Distances.Euclidean(0.0)), Bool[1, 0, 0])
```
"""
struct DiscreteKernel{T} <: Kernel where {
    T<:Union{Nothing, AbstractArray{Bool}}
}
    kernel::Kernel
    dims::T
end
DiscreteKernel(kernel::Kernel) = DiscreteKernel(kernel, nothing)

function (dk::DiscreteKernel)(x1, x2)
    r = discrete_round(dk.dims)
    dk.kernel(r(x1), r(x2))
end
(dk::DiscreteKernel)(x1::ForwardDiff.Dual, x2) = dk(x1.value, x2)
(dk::DiscreteKernel)(x1, x2::ForwardDiff.Dual) = dk(x1, x2.value)
(dk::DiscreteKernel)(x1::ForwardDiff.Dual, x2::ForwardDiff.Dual) = dk(x1.value, x2.value)

discrete_round() = x -> round.(x)
discrete_round(::Nothing) = discrete_round()
discrete_round(dims::AbstractArray{<:Bool}) = x -> cond_round.(x, dims)

function KernelFunctions.with_lengthscale(dk::DiscreteKernel, lengthscale::Real)
    return DiscreteKernel(with_lengthscale(dk.kernel, lengthscale), dk.dims)
end
function KernelFunctions.with_lengthscale(dk::DiscreteKernel, lengthscales::AbstractVector{<:Real})
    return DiscreteKernel(with_lengthscale(dk.kernel, lengthscales), dk.dims)
end

model_posterior(model::Nonparametric, data::ExperimentDataMLE) =
    model_posterior(model, data.X, data.Y, data.length_scales, data.noise_vars)

model_posterior(model::Nonparametric, data::ExperimentDataBI) =
    model_posterior.(Ref(model), Ref(data.X), Ref(data.Y), data.length_scales, eachcol(data.noise_vars))

function model_posterior(
    model::Nonparametric,
    X::AbstractMatrix{NUM},
    Y::AbstractMatrix{NUM},
    length_scales::AbstractMatrix{NUM},
    noise_vars::AbstractArray{NUM},
) where {NUM<:Real}
    finite_gps = finite_gp.(Ref(X), Ref(model.mean), Ref(model.kernel), eachcol(length_scales), noise_vars)
    posterior_gps = posterior.(finite_gps, eachrow(Y))

    function posterior(x)
        in = [x]
        mean = Vector{NUM}(undef, length(posterior_gps))
        var = Vector{NUM}(undef, length(posterior_gps))
        
        for i in eachindex(posterior_gps)
            m, v = mean_and_var(posterior_gps[i](in))
            mean[i] = m[1]
            var[i] = v[1]
        end

        return mean, var
    end
end

function finite_gp(
    X::AbstractMatrix{NUM},
    mean::Nothing,
    kernel::Kernel,
    length_scales::AbstractArray{NUM},
    noise_var::NUM;
    min_param_val<:Real=MIN_PARAM_VALUE,
    min_noise<:Real=MIN_PARAM_VALUE,
) where {NUM<:Real}
    # for numerical stability
    params = length_scales .+ min_param_val
    noise = noise_var + min_noise

    kernel = with_lengthscale(kernel, params)
    GP(kernel)(X, noise)
end
function finite_gp(
    X::AbstractMatrix{NUM},
    mean::Base.Callable,
    kernel::Kernel,
    length_scales::AbstractArray{NUM},
    noise_var::NUM;
    min_param_val<:Real=MIN_PARAM_VALUE,
    min_noise<:Real=MIN_PARAM_VALUE,
) where {NUM<:Real}
    # for numerical stability
    params = length_scales .+ min_param_val
    noise = noise_var + min_noise

    kernel = with_lengthscale(kernel, params)
    GP(CustomMean(mean), kernel)(X, noise)
end

# Log-likelihood of GP hyperparameters and noise variance.
function model_loglike(model::Nonparametric, noise_var_prior, data::AbstractExperimentData)
    params_loglike = model_params_loglike(model, data.X, data.Y)
    loglike(length_scales, noise_vars) = params_loglike(length_scales, noise_vars) + logpdf(noise_var_prior, noise_vars)
end

# Log-likelihood of GP hyperparameters.
function model_params_loglike(model::Nonparametric, X::AbstractMatrix{NUM}, Y::AbstractMatrix{NUM}) where {NUM<:Real}
    function params_loglike(length_scales, noise_vars)
        function ll_data_dim(X, y, length_scales, noise_var)
            gp = finite_gp(X, mean, kernel, length_scales, noise_var)
            logpdf(gp, y)
        end

        ll_data = mapreduce(p -> ll_data_dim(X, p...), +, zip(eachrow(Y), eachcol(length_scales), noise_vars))
        ll_length_scales = mapreduce(p -> logpdf(p...), +, zip(model.length_scale_priors, eachcol(length_scales)))
        ll_data + ll_length_scales
    end
end
