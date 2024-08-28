
"""
The minimal value of length scales and noise deviation used for GPs to avoid numerical issues.
"""
const MIN_PARAM_VALUE = 1e-8

"""
    GaussianProcess(; kwargs...)

A Gaussian Process surrogate model. Each output dimension is modeled by a separate independent process.

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

sliceable(::GaussianProcess) = true

function slice(m::GaussianProcess, idx::Int)
    return GaussianProcess(
        mean_slice(m.mean, idx),
        m.kernel,
        [m.amp_priors[idx]],
        [m.length_scale_priors[idx]],
    )
end

mean_slice(mean::Nothing, idx::Int) = nothing
mean_slice(mean::Function, idx::Int) = x -> mean(x)[idx]

θ_slice(m::GaussianProcess, idx::Int) = nothing

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

function model_posterior(model::GaussianProcess, data::ExperimentDataMAP)
    slices = model_posterior_slice.(Ref(model), Ref(data), 1:y_dim(data))

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
    post_gp = posterior_gp(model, data, slice)
    
    function post(x::AbstractVector{<:Real})
        μ, std = post(hcat(x))
        return μ[1], std[1]
    end
    function post(X::AbstractMatrix{<:Real})
        μ, var = mean_and_var(post_gp(X))
        return μ, sqrt.(var)
    end
    return post
end

"""
Construct posterior GP for a given `y` dimension via the AbstractGPs.jl library.
"""
function posterior_gp(model::GaussianProcess, data::ExperimentDataMAP, slice::Int)
    return AbstractGPs.posterior(
        finite_gp(
            data.X,
            mean_slice(model.mean, slice),
            model.kernel,
            data.length_scales[:,slice],
            data.amplitudes[slice],
            data.noise_std[slice],
        ),
        data.Y[slice,:],
    )
end

"""
Construct finite GP via the AbstractGPs.jl library.
"""
function finite_gp(
    X::AbstractMatrix{<:Real},
    mean::Union{Nothing, Function},
    kernel::Kernel,
    length_scales::AbstractVector{<:Real},
    amplitude::Real,
    noise_std::Real;
    min_param_val::Real=MIN_PARAM_VALUE,
)
    # for numerical stability
    length_scales = max.(length_scales, min_param_val)
    noise_std = max(noise_std, min_param_val)

    kernel = (amplitude^2) * with_lengthscale(kernel, length_scales)
    return finite_gp_(X, mean, kernel, noise_std)
end

finite_gp_(X::AbstractMatrix{<:Real}, mean::Nothing, kernel::Kernel, noise_std::Real) = GP(kernel)(X, noise_std^2)
finite_gp_(X::AbstractMatrix{<:Real}, mean::Function, kernel::Kernel, noise_std::Real) = GP(mean, kernel)(X, noise_std^2)

function model_loglike(model::GaussianProcess, noise_std_priors::AbstractVector{<:UnivariateDistribution}, data::ExperimentData)
    function loglike(θ, λ, α, noise_std)
        ll_data = data_loglike(model, data.X, data.Y, λ, α, noise_std)
        ll_params = model_params_loglike(model, λ, α)
        ll_noise = noise_loglike(noise_std_priors, noise_std)
        return ll_data + ll_params + ll_noise
    end
end

function data_loglike(
    model::GaussianProcess,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    λ::AbstractMatrix{<:Real},
    α::AbstractVector{<:Real},
    noise_std::AbstractVector{<:Real},
)
    y_dim = size(Y)[1]
    means = mean_slice.(Ref(model.mean), 1:y_dim)

    return sum(data_loglike_slice.(Ref(X), eachrow(Y), means, Ref(model.kernel), eachcol(λ), α, noise_std))
end

function data_loglike_slice(
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real},
    mean::Union{Nothing, Function},
    kernel::Kernel,
    length_scales::AbstractVector{<:Real},
    amplitude::Real,
    noise_std::Real,
)
    gp = finite_gp(X, mean, kernel, length_scales, amplitude, noise_std)
    return logpdf(gp, y)
end

function model_params_loglike(model::GaussianProcess, λ::AbstractMatrix{<:Real}, α::AbstractVector{<:Real})
    ll_λ = sum(logpdf.(model.length_scale_priors, eachcol(λ)))
    ll_α = sum(logpdf.(model.amp_priors, α))
    return ll_λ + ll_α
end

function sample_params(model::GaussianProcess)
    θ = Real[]
    λ = reduce(hcat, rand.(model.length_scale_priors))
    α = rand.(model.amp_priors)
    return θ, λ, α
end

param_priors(model::GaussianProcess) =
    UnivariateDistribution[], model.length_scale_priors, model.amp_priors
