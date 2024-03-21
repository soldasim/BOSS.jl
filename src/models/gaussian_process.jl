using AbstractGPs
using Distributions

"""
The minimal value of length scales and noise variance used for GPs to avoid numerical issues.
"""
const MIN_PARAM_VALUE = 1e-8

"""
    GaussianProcess(; kwargs...)

A Gaussian Process surrogate model.

# Keywords
- `mean::Union{Nothing, Function}`: Used as the mean function for the GP.
        Defaults to `nothing` equivalent to `x -> [0.]`.
- `kernel::Kernel`: The kernel used in the GP. Defaults to the `Matern52Kernel()`.
- `length_scale_priors::AbstractVector{<:MultivariateDistribution}`: The prior distributions
        for the length scales of the GP. The `length_scale_priors` should be a vector
        of `y_dim` `x_dim`-variate distributions where `x_dim` and `y_dim` are
        the dimensions of the input and output of the model respectively.
"""
struct GaussianProcess{
    M<:Union{Nothing, Function},
    P<:AbstractVector{<:MultivariateDistribution},
} <: SurrogateModel
    mean::M
    kernel::Kernel
    length_scale_priors::P
end
GaussianProcess(;
    mean=nothing,
    kernel=Matern32Kernel(),
    length_scale_priors,
) = GaussianProcess(mean, kernel, length_scale_priors)

add_mean(m::GaussianProcess{Nothing}, mean::Function) =
    Nonparametric(mean, m.kernel, m.length_scale_priors)

# Workaround: https://discourse.julialang.org/t/zygote-gradient-does-not-work-with-abstractgps-custommean/87815/7
AbstractGPs.mean_vector(m::AbstractGPs.CustomMean, x::ColVecs) = map(m.f, eachcol(x.X))
AbstractGPs.mean_vector(m::AbstractGPs.CustomMean, x::RowVecs) = map(m.f, eachrow(x.X))

"""
    DiscreteKernel(kernel::Kernel, dims::AbstractVector{Bool})
    DiscreteKernel(kernel::Kernel)

A kernel for dealing with discrete variables.
It is used as a wrapper around any other `AbstractGPs.Kernel`.

The field `dims` can be used to specify only some dimension as discrete.
All dimensions are considered as discrete if `dims` is not provided.

This structure is used internally by the BOSS algorithm.
The end user of BOSS.jl is not expected to use this structure.
Use the `BOSS.Domain` passed to the `BOSS.BossProblem`
to define discrete dimensions instead.

See also: `BOSS.BossProblem`(@ref)

# Examples:
```julia-repl
julia> DiscreteKernel(Matern52Kernel())
DiscreteKernel{Missing}(Matern 5/2 Kernel (metric = Distances.Euclidean(0.0)), missing)

julia> DiscreteKernel(Matern52Kernel(), [true, false, false])
DiscreteKernel{Vector{Bool}}(Matern 5/2 Kernel (metric = Distances.Euclidean(0.0)), Bool[1, 0, 0])
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

KernelFunctions.with_lengthscale(dk::DiscreteKernel, lengthscale::Real) =
    DiscreteKernel(with_lengthscale(dk.kernel, lengthscale), dk.dims)
KernelFunctions.with_lengthscale(dk::DiscreteKernel, lengthscales::AbstractVector{<:Real}) =
    DiscreteKernel(with_lengthscale(dk.kernel, lengthscales), dk.dims)

# Necessary to make `DiscreteKernel` work with ForwardDiff.jl.
# See: https://github.com/Sheld5/BOSS.jl/issues/4
KernelFunctions.kernelmatrix_diag(dk::DiscreteKernel, x::AbstractVector) =
    kernelmatrix_diag(dk.kernel, discrete_round.(Ref(dk.dims), x))

make_discrete(m::GaussianProcess, discrete::AbstractVector{<:Bool}) =
    Nonparametric(m.mean, make_discrete(m.kernel, discrete), m.length_scale_priors)

make_discrete(k::Kernel, discrete::AbstractVector{<:Bool}) = DiscreteKernel(k, discrete)
make_discrete(k::DiscreteKernel, discrete::AbstractVector{<:Bool}) = make_discrete(k.kernel, discrete)

model_posterior(model::GaussianProcess, data::ExperimentDataMLE) =
    model_posterior(model, data.X, data.Y, data.length_scales, data.noise_vars)

model_posterior(model::GaussianProcess, data::ExperimentDataBI) =
    model_posterior.(Ref(model), Ref(data.X), Ref(data.Y), data.length_scales, eachcol(data.noise_vars))

function model_posterior(
    model::GaussianProcess,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    length_scales::AbstractMatrix{<:Real},
    noise_vars::AbstractVector{<:Real},
)
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
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real},
    length_scales::AbstractVector{<:Real},
    noise_var::Real,
)
    posterior_gp = AbstractGPs.posterior(finite_gp(mean, kernel, X, length_scales, noise_var), y)
    return (x) -> first.(mean_and_var(posterior_gp(hcat(x))))
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
)
    # for numerical stability
    length_scales = max.(length_scales, min_param_val)
    noise_var = max(noise_var, min_param_val)

    kernel = with_lengthscale(kernel, length_scales)
    return finite_gp_(mean, kernel, X, noise_var)
end

finite_gp_(::Nothing, kernel::Kernel, X::AbstractMatrix{<:Real}, noise_var::Real) = GP(kernel)(X, noise_var)
finite_gp_(mean::Function, kernel::Kernel, X::AbstractMatrix{<:Real}, noise_var::Real) = GP(mean, kernel)(X, noise_var)

function model_loglike(model::GaussianProcess, noise_var_priors::AbstractVector{<:UnivariateDistribution}, data::ExperimentData)
    function loglike(θ, length_scales, noise_vars)
        ll_params = model_params_loglike(model, length_scales)
        ll_data = model_data_loglike(model, length_scales, noise_vars, data.X, data.Y)
        ll_noise = noise_loglike(noise_var_priors, noise_vars)
        return ll_params + ll_data + ll_noise
    end
end

function model_params_loglike(model::GaussianProcess, length_scales::AbstractMatrix{<:Real})
    return mapreduce(p -> logpdf(p...), +, zip(model.length_scale_priors, eachcol(length_scales)))
end

function model_data_loglike(
    model::GaussianProcess,
    length_scales::AbstractMatrix{<:Real},
    noise_vars::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    y_dim = size(Y)[1]
    means = isnothing(model.mean) ? fill(nothing, y_dim) : [x->model.mean(x)[i] for i in 1:y_dim]
    function ll_data_dim(X, y, mean, length_scales, noise_var)
        gp = finite_gp(mean, model.kernel, X, length_scales, noise_var)
        return logpdf(gp, y)
    end
    return mapreduce(p -> ll_data_dim(X, p...), +, zip(eachrow(Y), means, eachcol(length_scales), noise_vars))
end

function sample_params(model::GaussianProcess, noise_var_priors::AbstractVector{<:UnivariateDistribution})
    θ = Real[]
    λ = reduce(hcat, rand.(model.length_scale_priors))
    noise_vars = rand.(noise_var_priors)
    return θ, λ, noise_vars
end

param_priors(model::GaussianProcess) =
    UnivariateDistribution[], model.length_scale_priors
