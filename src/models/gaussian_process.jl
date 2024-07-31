
"""
The minimal value of length scales and noise deviation used for GPs to avoid numerical issues.
"""
const MIN_PARAM_VALUE = 1e-8

"""
    GaussianProcess(; kwargs...)

A Gaussian Process surrogate model.

# Keywords
- `mean::Union{Nothing, Function}`: Used as the mean function for the GP.
        Defaults to `nothing` equivalent to `x -> [0.]`.
- `kernel::Kernel`: The kernel used in the GP. Defaults to the `Matern32Kernel()`.
- `amp_priors::AbstractVector{<:UnivariateDistribution}`: The prior distributions
        for the amplitude hyperparameters of the GP. The `amp_priors` should be a vector
        of `y_dim` univariate distributions.
- `length_scale_priors::AbstractVector{<:MultivariateDistribution}`: The prior distributions
        for the length scales of the GP. The `length_scale_priors` should be a vector
        of `y_dim` `x_dim`-variate distributions where `x_dim` and `y_dim` are
        the dimensions of the input and output of the model respectively.
"""
struct GaussianProcess{
    M<:Union{Nothing, Function},
    S<:AbstractVector{<:UnivariateDistribution},
    L<:AbstractVector{<:MultivariateDistribution},
} <: SurrogateModel
    mean::M
    kernel::Kernel
    amp_priors::S
    length_scale_priors::L
end
GaussianProcess(;
    mean=nothing,
    kernel=Matern32Kernel(),
    amp_priors,
    length_scale_priors,
) = GaussianProcess(mean, kernel, amp_priors, length_scale_priors)

add_mean(m::GaussianProcess{Nothing}, mean::Function) =
    Nonparametric(mean, m.kernel, m.amp_priors, m.length_scale_priors)

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
Use the `Domain` passed to the `BossProblem`
to define discrete dimensions instead.

See also: `BossProblem`(@ref)

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
    Nonparametric(m.mean, make_discrete(m.kernel, discrete), m.amp_priors, m.length_scale_priors)

make_discrete(k::Kernel, discrete::AbstractVector{<:Bool}) = DiscreteKernel(k, discrete)
make_discrete(k::DiscreteKernel, discrete::AbstractVector{<:Bool}) = make_discrete(k.kernel, discrete)

model_posterior(model::GaussianProcess, data::ExperimentDataMAP) =
    model_posterior(model, data.X, data.Y, data.length_scales, data.amplitudes, data.noise_std)

function model_posterior(
    model::GaussianProcess,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    λ::AbstractMatrix{<:Real},
    α::AbstractVector{<:Real},
    noise_std::AbstractVector{<:Real},
)
    y_dim = length(noise_std)
    means = isnothing(model.mean) ? fill(nothing, y_dim) : [x->model.mean(x)[i] for i in 1:y_dim]
    slices = model_posterior_slice.(means, Ref(model.kernel), Ref(X), eachrow(Y), eachcol(λ), α, noise_std)
    
    function post(x::AbstractVector{<:Real})
        μ, std = post(hcat(x))
        return μ[:,1], std[:,1]
    end
    function post(x::AbstractMatrix{<:Real})
        preds = map(p->p(x), slices)
        μ = mapreduce(pred -> pred[1]', vcat, preds)
        std = mapreduce(pred -> pred[2]', vcat, preds)
        return μ, std
    end
    return post
end

function model_posterior_slice(model::GaussianProcess, data::ExperimentDataMAP, slice::Int)
    return model_posterior_slice(
        isnothing(model.mean) ? nothing : (x -> model.mean(x)[slice]),
        model.kernel,
        data.X,
        data.Y[slice,:],
        data.length_scales[:,slice],
        data.amplitudes[slice],
        data.noise_std[slice],
    )
end

function model_posterior_slice(
    mean::Union{Nothing, Function},
    kernel::Kernel,
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real},
    λ::AbstractVector{<:Real},
    α::Real,
    noise_std::Real,
)
    posterior_gp = AbstractGPs.posterior(finite_gp(mean, kernel, X, λ, α, noise_std), y)
    
    function post(x::AbstractVector{<:Real})
        μ, std = post(hcat(x))
        return μ[1], std[1]
    end
    function post(X::AbstractMatrix{<:Real})
        μ, var = mean_and_var(posterior_gp(X))
        return μ, sqrt.(var)
    end
    return post
end

"""
Construct a finite GP via the AbstractGPs.jl library.
"""
function finite_gp(
    mean::Union{Nothing, Function},
    kernel::Kernel,
    X::AbstractMatrix{<:Real},
    length_scales::AbstractVector{<:Real},
    amplitude::Real,
    noise_std::Real;
    min_param_val::Real=MIN_PARAM_VALUE,
)
    # for numerical stability
    length_scales = max.(length_scales, min_param_val)
    noise_std = max(noise_std, min_param_val)

    kernel = (amplitude^2) * with_lengthscale(kernel, length_scales)
    return finite_gp_(mean, kernel, X, noise_std)
end

finite_gp_(::Nothing, kernel::Kernel, X::AbstractMatrix{<:Real}, noise_std::Real) = GP(kernel)(X, noise_std^2)
finite_gp_(mean::Function, kernel::Kernel, X::AbstractMatrix{<:Real}, noise_std::Real) = GP(mean, kernel)(X, noise_std^2)

function model_loglike(model::GaussianProcess, noise_std_priors::AbstractVector{<:UnivariateDistribution}, data::ExperimentData)
    function loglike(θ, λ, α, noise_std)
        ll_params = model_params_loglike(model, λ, α)
        ll_data = model_data_loglike(model, λ, α, noise_std, data.X, data.Y)
        ll_noise = noise_loglike(noise_std_priors, noise_std)
        return ll_params + ll_data + ll_noise
    end
end

function model_params_loglike(model::GaussianProcess, λ::AbstractMatrix{<:Real}, α::AbstractVector{<:Real})
    ll_amplitudes = mapreduce(p -> logpdf(p...), +, zip(model.amp_priors, α))
    ll_length_scales = mapreduce(p -> logpdf(p...), +, zip(model.length_scale_priors, eachcol(λ)))
    return ll_amplitudes + ll_length_scales
end

function model_data_loglike(
    model::GaussianProcess,
    λ::AbstractMatrix{<:Real},
    α::AbstractVector{<:Real},
    noise_std::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    y_dim = size(Y)[1]
    means = isnothing(model.mean) ? fill(nothing, y_dim) : [x->model.mean(x)[i] for i in 1:y_dim]
    function ll_data_dim(X, y, mean, length_scales, amp, noise_std)
        gp = finite_gp(mean, model.kernel, X, length_scales, amp, noise_std)
        return logpdf(gp, y)
    end
    return mapreduce(p -> ll_data_dim(X, p...), +, zip(eachrow(Y), means, eachcol(λ), α, noise_std))
end

function sample_params(model::GaussianProcess)
    θ = Real[]
    λ = reduce(hcat, rand.(model.length_scale_priors))
    α = rand.(model.amp_priors)
    return θ, λ, α
end

param_priors(model::GaussianProcess) =
    UnivariateDistribution[], model.length_scale_priors, model.amp_priors
