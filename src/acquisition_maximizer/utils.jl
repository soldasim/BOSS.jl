using LatinHypercubeSampling
using Distributions


# - - - GENERATING OPTIMIZATION STARTING POINTS - - - - -

"""
    random_start(bounds) -> x

Return a random point form the given bounds.
"""
function random_start(bounds::AbstractBounds)
    lb, ub = bounds
    dim = length(lb)
    start = rand(dim) .* (ub .- lb) .+ lb
    return start
end

"""
    generate_starts_LHC(bounds, count) -> X

Return a matrix of latin hyper-cube vertices in the given bounds.
"""
function generate_starts_LHC(bounds::AbstractBounds, count::Int)
    @assert count > 1  # `randomLHC(count, dim)` returns NaNs if `count == 1`
    lb, ub = bounds
    x_dim = length(lb)
    starts = scaleLHC(randomLHC(count, x_dim), [(lb[i], ub[i]) for i in 1:x_dim]) |> transpose
    return starts
end

# # TODO: unused
# """
# Moves the points to the interior of the given bounds.
# """
# function move_to_interior!(points::AbstractMatrix{<:Float64}, bounds::AbstractBounds; gap=0.)
#     for dim in size(points)[1]
#         points[dim,:] .= move_to_interior.(points[dim,:], Ref((bounds[1][dim], bounds[2][dim])); gap)
#     end
#     return points
# end
# function move_to_interior!(point::AbstractVector{<:Float64}, bounds::AbstractBounds; gap=0.)
#     dim = length(point)
#     point .= move_to_interior.(point, ((bounds[1][i], bounds[2][i]) for i in 1:dim); gap)
# end
# function move_to_interior(point::Float64, bounds::Tuple{<:Float64, <:Float64}; gap=0.)
#     (bounds[2] - point >= gap) || (point = bounds[2] - gap)
#     (point - bounds[1] >= gap) || (point = bounds[1] + gap)
#     return point
# end

"""
Sample all model parameters and noise variances from their respective prior distributions.
"""
function sample_params(model::Parametric, noise_var_priors::AbstractVector{<:UnivariateDistribution})
    θ = rand.(model.param_priors)
    λ = Float64[;;]
    noise_vars = rand.(noise_var_priors)
    return (θ=θ, λ=λ, noise_vars=noise_vars)
end
function sample_params(model::Nonparametric, noise_var_priors::AbstractVector{<:UnivariateDistribution})
    θ = Float64[]
    λ = reduce(hcat, rand.(model.length_scale_priors))
    noise_vars = rand.(noise_var_priors)
    return (θ=θ, λ=λ, noise_vars=noise_vars)
end
function sample_params(model::Semiparametric, noise_var_priors::AbstractVector{<:UnivariateDistribution})
    θ = rand.(model.parametric.param_priors)
    λ = reduce(hcat, rand.(model.nonparametric.length_scale_priors))
    noise_vars = rand.(noise_var_priors)
    return (θ=θ, λ=λ, noise_vars=noise_vars)
end


# - - - PARAMETER TRANSFORMATIONS - - - - -

"""
Transform all model parameters and noise variances into one parameter vector.
""" 
function vectorize_params(
    θ::AbstractVector{<:Real}, λ::AbstractMatrix{<:Real}, noise_vars::AbstractVector{<:Real},
    activation_function::Function,
    activation_mask::AbstractVector{Bool},
    skip_mask::AbstractVector{Bool},
)
    params = vectorize_params(θ, λ, noise_vars)
    params .= cond_func(inverse(activation_function)).(activation_mask, params)
    params = params[skip_mask]
    return params
end

vectorize_params(θ::AbstractVector{<:Real}, λ::AbstractMatrix{<:Real}, noise_vars::AbstractVector{<:Real}) =
    vcat(θ, reduce(vcat, eachcol(λ); init=eltype(λ)[]), noise_vars)

"""
Transform vectorized model parameters back into separate vectors/matrices.
"""
function devectorize_params(
    model::SurrogateModel,
    params::AbstractVector{<:Real},
    activation_function::Function,
    activation_mask::AbstractVector{Bool},
    skipped_values::AbstractVector{<:Real},
    skip_mask::AbstractVector{Bool},
)
    # @assert length(skipped_values) == length(skip_mask)
    params_ = params
    params = similar(params, length(skip_mask))   
    params[skip_mask] .= params_
    params[.!skip_mask] .= skipped_values

    params .= cond_func(activation_function).(activation_mask .& skip_mask, params)

    return devectorize_params(model, params)
end

function devectorize_params(model::Parametric, params::AbstractVector{<:Real})
    θ_len_ = θ_len(model)
    
    θ = params[1:θ_len_]
    noise_vars = params[θ_len_+1:end]
    return (θ=θ, noise_vars=noise_vars)
end
function devectorize_params(model::Nonparametric, params::AbstractVector{<:Real})
    x_dim_, y_dim_, λ_len_ = x_dim(model), y_dim(model), λ_len(model)
    
    length_scales = reshape(params[1:λ_len_], x_dim_, y_dim_)
    noise_vars = params[λ_len_+1:end]
    return (length_scales=length_scales, noise_vars=noise_vars)
end
function devectorize_params(model::Semiparametric, params::AbstractVector{<:Real})
    x_dim_, y_dim_, θ_len_, λ_len_ = x_dim(model), y_dim(model), θ_len(model), λ_len(model)
    
    θ = params[1:θ_len_]
    length_scales = reshape(params[θ_len_+1:θ_len_+λ_len_], x_dim_, y_dim_)
    noise_vars = params[θ_len_+λ_len_+1:end]
    return (θ=θ, length_scales=length_scales, noise_vars=noise_vars)
end

"""
Create a binary mask describing to which positions of the vectorized parameters
will the activation function be applied.

The activation function will be applited to all noise variances and GP length scales
if `mask_noisevar_and_lengthscales=true`.

Use the binary vector `mask_theta` to define to which model parameters
will the activation function be applied as well.
"""
create_activation_mask(
    problem::OptimizationProblem,
    mask_noisevar_and_lengthscales::Bool,
    mask_theta::Union{<:AbstractVector{Bool}, Nothing},
) = create_activation_mask(param_count(problem.model) + y_dim(problem), θ_len(problem.model), mask_noisevar_and_lengthscales, mask_theta)

function create_activation_mask(
    params_total::Int,
    θ_len::Int,
    mask_noisevar_and_lengthscales::Bool,
    mask_theta::Union{<:AbstractVector{Bool}, Nothing},
)
    mask = fill(false, params_total)
    if !isnothing(mask_theta)
        mask[1:θ_len] .= mask_theta
    end
    if mask_noisevar_and_lengthscales
        mask[θ_len+1:end] .= true
    end
    return mask
end

create_dirac_skip_mask(problem::OptimizationProblem) =
    create_dirac_skip_mask(problem.model, problem.noise_var_priors)

create_dirac_skip_mask(model::Parametric, noise_var_priors::AbstractVector{<:UnivariateDistribution}) =
    create_dirac_skip_mask(model.param_priors, MultivariateDistribution[], noise_var_priors)
create_dirac_skip_mask(model::Nonparametric, noise_var_priors::AbstractVector{<:UnivariateDistribution}) =
    create_dirac_skip_mask(UnivariateDistribution[], model.length_scale_priors, noise_var_priors)
create_dirac_skip_mask(model::Semiparametric, noise_var_priors::AbstractVector{<:UnivariateDistribution}) =
    create_dirac_skip_mask(model.parametric.param_priors, model.nonparametric.length_scale_priors, noise_var_priors)

"""
Create a binary mask to skip all parameters with Dirac priors
from the optimization parameters.

Return the binary skip mask and a vector of the skipped Dirac values.
"""
function create_dirac_skip_mask(
    θ_priors::AbstractVector{<:UnivariateDistribution},
    λ_priors::AbstractVector{<:MultivariateDistribution},
    noise_var_priors::AbstractVector{<:UnivariateDistribution},
)    
    priors = Any[]
    # θ
    append!(priors, θ_priors)
    # λ
    for md in λ_priors
        if md isa Product
            append!(priors, md.v)
        else
            append!(priors, (nothing for _ in 1:length(md)))
        end
    end
    # noise_vars
    append!(priors, noise_var_priors)
    
    params_total = length(priors)
    skip_mask = fill(true, params_total)
    skipped_values = Float64[]

    for i in eachindex(priors)
        if priors[i] isa Dirac
            skip_mask[i] = false
            push!(skipped_values, priors[i].value)
        end
    end

    return skip_mask, skipped_values
end

# Activation function to ensure positive optimization arguments.
softplus(x) = log(one(x) + exp(x))
inv_softplus(x) = log(exp(x) - one(x))

inverse(::typeof(softplus)) = inv_softplus
inverse(::typeof(inv_softplus)) = softplus
