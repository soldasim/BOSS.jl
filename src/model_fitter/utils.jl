
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

function vectorize_params(θ::AbstractVector{<:Real}, λ::AbstractMatrix{<:Real}, noise_vars::AbstractVector{<:Real})
    λ = vectorize_length_scales(λ)
    
    # skip empty params for type stability
    nonempty_params = filter(!isempty, (θ, λ, noise_vars))
    isempty(nonempty_params) && return promote_type(eltype.((θ, λ, noise_vars))...)[]

    params = reduce(vcat, nonempty_params)
    return params
end

vectorize_length_scales(λ::AbstractMatrix) =
    reduce(vcat, eachcol(λ); init=eltype(λ)[])

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

function devectorize_params(model::SurrogateModel, params::AbstractVector{<:Real})
    θ_shape, λ_shape = param_shapes(model)
    θ_len, λ_len = prod(θ_shape), prod(λ_shape)

    θ = params[1:θ_len]
    λ = reshape(params[θ_len+1:θ_len+λ_len], λ_shape)
    noise_vars = params[θ_len+λ_len+1:end]
    
    return θ, λ, noise_vars
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
    problem::BossProblem,
    mask_noisevar_and_lengthscales::Bool,
    mask_theta::Union{<:AbstractVector{Bool}, Nothing},
) = create_activation_mask(
    params_total(problem),
    param_counts(problem.model)[1],
    mask_noisevar_and_lengthscales,
    mask_theta,
)

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

create_dirac_skip_mask(problem::BossProblem) =
    create_dirac_skip_mask(problem.model, problem.noise_var_priors)

create_dirac_skip_mask(model::SurrogateModel, noise_var_priors::AbstractVector{<:UnivariateDistribution}) =
    create_dirac_skip_mask(param_priors(model)..., noise_var_priors)

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
