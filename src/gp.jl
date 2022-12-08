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
It can be used as a wrapper around any other `AbstractGPs.Kernel`.

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

cond_round(x, b::Bool) = b ? round(x) : x

function KernelFunctions.with_lengthscale(dk::DiscreteKernel, lengthscale::Real)
    return DiscreteKernel(with_lengthscale(dk.kernel, lengthscale), dk.dims)
end
function KernelFunctions.with_lengthscale(dk::DiscreteKernel, lengthscales::AbstractVector{<:Real})
    return DiscreteKernel(with_lengthscale(dk.kernel, lengthscales), dk.dims)
end

gp_param_count(x_dim) = x_dim

function gp_model(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, params, noise, mean::Nothing, kernel::Kernel)
    finite_gps = construct_finite_gp.(Ref(X), params, noise, Ref(nothing), Ref(kernel))
    gp_model(finite_gps, Y)
end
function gp_model(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, params, noise, mean, kernel::Kernel)
    means = [x -> mean(x)[i] for i in 1:length(params)]
    gp_model(X, Y, params, noise, means, kernel)
end
function gp_model(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, params, noise, means::AbstractArray, kernel::Kernel)
    finite_gps = construct_finite_gp.(Ref(X), params, noise, means, Ref(kernel))
    gp_model(finite_gps, Y)
end
function gp_model(finite_gps::AbstractArray{<:AbstractGPs.FiniteGP}, Y::AbstractMatrix{<:Real})
    posts = posterior.(finite_gps, eachrow(Y))

    function model(x)
        in = [x]
        mean = Vector{Float64}(undef, length(posts))
        var = Vector{Float64}(undef, length(posts))
        
        for i in eachindex(posts)
            m, v = mean_and_var(posts[i](in))
            mean[i] = m[1]
            var[i] = v[1]
        end

        return mean, var
    end
end

function construct_finite_gp(X::AbstractMatrix{<:Real}, params, noise, mean::Nothing, kernel::Kernel; min_param_val=MIN_PARAM_VALUE, min_noise=MIN_PARAM_VALUE)
    # for numerical stability
    params = params .+ min_param_val
    noise = noise + min_noise

    kernel = with_lengthscale(kernel, params)
    GP(kernel)(X, noise)
end
function construct_finite_gp(X::AbstractMatrix{<:Real}, params, noise, mean, kernel::Kernel; min_param_val=MIN_PARAM_VALUE, min_noise=MIN_PARAM_VALUE)
    # for numerical stability
    params = params .+ min_param_val
    noise = noise + min_noise

    kernel = with_lengthscale(kernel, params)
    GP(CustomMean(mean), kernel)(X, noise)
end

function gp_params_loglike(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, params_prior, mean, kernel::Kernel)
    function ll_data(params, noise)
        gp = construct_finite_gp(X, params, noise, mean, kernel)
        logpdf(gp, y)
    end

    function loglike(params, noise)
        ll_data(params, noise) + logpdf(params_prior, params)
    end
end

function opt_gp_params(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, params_prior, noise_prior, mean, kernel::Kernel; multistart, optim_options=Optim.Options(), parallel, info=true, debug=false, min_param_value=MIN_PARAM_VALUE)
    softplus(x) = log(1. + exp(x))
    lift(p) = softplus.(p) .+ min_param_value  # 'min_param_value' for numerical stability
    
    params_loglike = gp_params_loglike(X, y, params_prior, mean, kernel)
    noise_loglike = noise -> logpdf(noise_prior, noise)

    function loglike(p)
        noise, params... = p
        return params_loglike(params, noise) + noise_loglike(noise)
    end

    starts = vcat(rand(noise_prior, multistart)', rand(params_prior, multistart))
    constraints = (
        vcat(minimum(noise_prior), minimum(params_prior)),
        vcat(maximum(noise_prior), maximum(params_prior))
    )

    p, _ = optim_Optim_multistart(p -> loglike(lift(p)), starts, constraints; parallel, options=optim_options, info, debug)
    noise, params... = lift(p)
    return params, noise
end

function opt_gps_params(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, params_priors, noise_priors, means, kernel::Kernel; multistart::Int, optim_options=Optim.Options(), parallel, info, debug)
    P = opt_gp_params.(Ref(X), eachrow(Y), params_priors, noise_priors, means, Ref(kernel); multistart, optim_options, parallel, info, debug)
    params = [p[1] for p in P]
    noise = [p[2] for p in P]
    return params, noise
end

Turing.@model function gp_turing_model(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, mean, kernel::Kernel, params_prior, noise_prior)
    params ~ params_prior
    noise ~ noise_prior
    
    gp = construct_finite_gp(X, params, noise, mean, kernel)
    
    y ~ gp
end

function sample_gp_params(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, params_prior, noise_prior, mean, kernel::Kernel; x_dim::Int, mc_settings::MCSettings, parallel)
    model = gp_turing_model(X, y, mean, kernel, params_prior, noise_prior)
    param_symbols = vcat([Symbol("params[$i]") for i in 1:gp_param_count(x_dim)], :noise)
    samples = sample_params_turing(model, param_symbols, mc_settings; parallel)
    
    params = reduce(vcat, transpose.(samples[1:gp_param_count(x_dim)]))
    noise = samples[end]
    return params, noise
end

function sample_gps_params(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, params_priors, noise_priors, means, kernel::Kernel; mc_settings::MCSettings, parallel)
    x_dim = size(X)[1]

    samples = sample_gp_params.(Ref(X), eachrow(Y), params_priors, noise_priors, means, Ref(kernel); x_dim, mc_settings, parallel)
    params = [s[1] for s in samples]
    noise = [s[2] for s in samples]
    return params, noise
end

function fit_nonparametric_model(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, kernel::Kernel, gp_params_priors, noise_priors; param_fit_alg, multistart=80, optim_options=Optim.Options(), mc_settings=MCSettings(400, 20, 8, 6), parallel=true, info=false, debug=false)
    if param_fit_alg == :MLE
        gp_params, noise = opt_gps_params(X, Y, gp_params_priors, noise_priors, nothing, kernel; multistart, optim_options, parallel, info, debug)
        nonparametric = gp_model(X, Y, gp_params, noise, nothing, kernel)

        model_samples = nothing
    
    elseif param_fit_alg == :BI
        gp_param_samples, noise_samples = sample_gps_params(X, Y, gp_params_priors, noise_priors, nothing, kernel; mc_settings, parallel)
        model_samples = [gp_model(X, Y, [s[:,i] for s in gp_param_samples], [s[i] for s in noise_samples], nothing, kernel) for i in 1:sample_count(mc_settings)]
        nonparametric = x -> (mapreduce(m -> m(x), .+, model_samples) ./ length(model_samples))  # for plotting only

        noise = noise_samples
        gp_params = gp_param_samples
    end

    return nonparametric, model_samples, gp_params, noise
end
