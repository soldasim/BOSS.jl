
"""
Transform all model parameters and noise deviations into one parameter vector.
"""
function vectorize_params(
    params::ModelParams,
    activation_function::Function,
    activation_mask::AbstractVector{Bool},
    skip_mask::AbstractVector{Bool},
)
    params = vectorize_params(params...)
    params .= cond_func(inverse(activation_function)).(activation_mask, params)
    params = params[skip_mask]
    return params
end

function vectorize_params(θ::AbstractVector{<:Real}, λ::AbstractMatrix{<:Real}, α::AbstractVector{<:Real}, noise_std::AbstractVector{<:Real})
    λ = vectorize_length_scales(λ)
    
    # skip empty params for type stability
    nonempty_params = filter(!isempty, (θ, λ, α, noise_std))
    isempty(nonempty_params) && return promote_type(eltype.((θ, λ, α, noise_std))...)[]

    params = reduce(vcat, nonempty_params)
    return params
end

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
    θ_shape, λ_shape, α_shape = param_shapes(model)
    θ_len, λ_len, α_len = prod.((θ_shape, λ_shape, α_shape))
    cumsums = [0, θ_len, θ_len + λ_len, θ_len + λ_len + α_len]

    θ, λ, α = (params[cumsums[i]+1:cumsums[i+1]] for i in 1:3)
    noise_std = params[cumsums[end]+1:end]

    λ = devectorize_length_scales(λ, λ_shape)

    return θ, λ, α, noise_std
end

vectorize_length_scales(λ::AbstractMatrix) =
    reduce(vcat, eachcol(λ); init=eltype(λ)[])

devectorize_length_scales(λ::AbstractVector, λ_shape::Tuple{<:Int, <:Int}) =
    reshape(λ, λ_shape)

"""
Create a binary mask describing to which positions of the vectorized parameters
will the activation function be applied.

The activation function will be applited to all noise deviations and GP hyperaparameters
if `mask_hyperparams = true`.

Use the binary vector `mask_theta` to define to which model parameters
will the activation function be applied as well.
"""
create_activation_mask(
    problem::BossProblem,
    mask_hyperparams::Bool,
    mask_params::Union{Bool, Vector{Bool}},
) = create_activation_mask(
    params_total(problem),
    param_counts(problem.model)[1],
    mask_hyperparams,
    mask_params,
)

function create_activation_mask(
    params_total::Int,
    θ_len::Int,
    mask_hyperparams::Bool,
    mask_params::Union{Bool, Vector{Bool}},
)
    return vcat(
        create_params_mask(mask_params, θ_len),
        fill(mask_hyperparams, params_total - θ_len),
    )
end

function create_params_mask(mask_params::Bool, θ_len::Int)
    return fill(mask_params, θ_len)
end
function create_params_mask(mask_params::Vector{Bool}, θ_len::Int)
    @assert length(mask_params) == θ_len 
    return mask_params
end

create_dirac_skip_mask(problem::BossProblem) =
    create_dirac_skip_mask(problem.model, problem.noise_std_priors)

create_dirac_skip_mask(model::SurrogateModel, noise_std_priors::AbstractVector{<:UnivariateDistribution}) =
    create_dirac_skip_mask(param_priors(model)..., noise_std_priors)

"""
Create a binary mask to skip all parameters with Dirac priors
from the optimization parameters.

Return the binary skip mask and a vector of the skipped Dirac values.
"""
function create_dirac_skip_mask(
    θ_priors::AbstractVector{<:UnivariateDistribution},
    λ_priors::AbstractVector{<:MultivariateDistribution},
    α_priors::AbstractVector{<:UnivariateDistribution},
    noise_std_priors::AbstractVector{<:UnivariateDistribution},
)
    priors = Union{Nothing, UnivariateDistribution}[]
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
    # α
    append!(priors, α_priors)
    # noise_std
    append!(priors, noise_std_priors)
    
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
