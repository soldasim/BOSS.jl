
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
    params::AbstractVector{<:Real},
    model::SurrogateModel,
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

    return devectorize_params(params, model)
end

function devectorize_params(params::AbstractVector{<:Real}, model::SurrogateModel)
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
function create_activation_mask(
    model::SurrogateModel,
    y_dim::Int,
    mask_theta::Union{Bool, Vector{Bool}},
    mask_hyperparams::Bool,
)
    return create_activation_mask(param_counts(model), y_dim, mask_theta, mask_hyperparams)
end

function create_activation_mask(
    param_counts::Tuple{<:Int, <:Int, <:Int},
    y_dim::Int,
    mask_theta::Union{Bool, Vector{Bool}},
    mask_hyperparams::Bool,
)
    θ_len, λ_len, α_len = param_counts
    return vcat(
        create_θ_mask(mask_theta, θ_len),
        fill(mask_hyperparams, λ_len + α_len + y_dim),
    )
end

function create_θ_mask(mask_params::Bool, θ_len::Int)
    return fill(mask_params, θ_len)
end
function create_θ_mask(mask_params::Vector{Bool}, θ_len::Int)
    @assert length(mask_params) == θ_len 
    return mask_params
end

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

# `return_all=false` version
function reduce_slice_results(results::AbstractVector{<:Tuple{<:ModelParams, <:Real}})
    params = reduce_slice_params(first.(results))
    loglike = sum(last.(results))
    return params, loglike
end

function reduce_slice_params(params::AbstractVector{<:ModelParams})
    θ = reduce(vcat, ith(1).(params))
    λ = reduce(hcat, ith(2).(params))
    α = reduce(vcat, ith(3).(params))
    noise_std = reduce(vcat, ith(4).(params))
    return θ, λ, α, noise_std
end

# `return_all=true` version
function reduce_slice_results(results::AbstractVector{<:Tuple{<:AbstractVector{<:ModelParams}, <:AbstractVector{<:Real}}})
    y_dim = length(results)
    sample_count = get_sample_count_(results)
    
    params = get_params_.(Ref(results), Ref(y_dim), 1:sample_count)
    loglikes = get_loglike_.(Ref(results), Ref(y_dim), 1:sample_count)
    return params, loglikes
end

function get_sample_count_(results::AbstractVector{<:Tuple{<:AbstractVector{<:ModelParams}, <:AbstractVector{<:Real}}})
    param_lens = length.(first.(results))
    loglike_lens = length.(last.(results))
    @assert all(param_lens .== loglike_lens)
    return minimum(param_lens)
end

function get_params_(results, y_dim::Int, idx::Int)
    params = [results[i][1][idx] for i in 1:y_dim]
    return reduce_slice_params(params)
end
function get_loglike_(results, y_dim::Int, idx::Int)
    loglikes = [results[i][2][idx] for i in 1:y_dim]
    return sum(loglikes)
end
