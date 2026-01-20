
"""
    InputTransform(forward)

Defines a transformation for input data.

## Fields
- `forward::Function`: A function `(x::AbstractVector{<:Real}) -> x_::AbstractVector{<:Real}`
        that transforms input points, or `(X::AbstractMatrix{<:Real}) -> X_::AbstractMatrix{<:Real}`
        for multiple points (columns).
"""
@kwdef struct InputTransform
    forward::Function
end

"""
    OutputTransform

Abstract supertype for output-space transformations.

Concrete variants:
- [`JointOutputTransform`](@ref): transforms the full output vector jointly.
- [`SlicedOutputTransform`](@ref): transforms each output dimension independently.
"""
abstract type OutputTransform end

"""
    JointOutputTransform(forward, backward)

Defines a joint bidirectional transformation for all output dimensions together.

`forward` transforms both the model-space output `y_` and its standard deviation `std_` to user space `(y, std)`.
`backward` maps user-space `y` back to model-space `y_`.

For a nonlinear transform, the forward function should propagate uncertainty appropriately.
For example, for `y = log(y_)`, the transform would be:
```julia
forward = (y_, std_) -> begin
    y = log.(y_)
    std = std_ ./ y_  # derivative: dy/dy_ = 1/y_
    return y, std
end
backward = y -> exp.(y)
```

## Fields
- `forward::Function`: A function `(y_::AbstractVector{<:Real}, std_::AbstractVector{<:Real}) -> (y::AbstractVector{<:Real}, std::AbstractVector{<:Real})`
        that transforms both mean and std from model to user space.
- `backward::Function`: A function `(y::AbstractVector{<:Real}) -> y_::AbstractVector{<:Real}`
        that transforms mean from user to model space.

See also: [`SlicedOutputTransform`](@ref)
"""
@kwdef struct JointOutputTransform <: OutputTransform
    forward::Function
    backward::Function
end

"""
    SlicedOutputTransform(forward, backward)

Defines a sliced bidirectional transformation where each output dimension is transformed independently.

Each `forward[i]` transforms both the model-space scalar `y_i_` and its standard deviation `std_i_` to user space `(y_i, std_i)`.
Each `backward[i]` maps user-space `y_i` back to model-space `y_i_`.

For a nonlinear transform, the forward function should propagate uncertainty appropriately.
For example, for `y_i = log(y_i_)`, the transform would be:
```julia
forward[i] = (y_i_, std_i_) -> begin
    y_i = log(y_i_)
    std_i = std_i_ / y_i_  # derivative: dy_i/dy_i_ = 1/y_i_
    return y_i, std_i
end
backward[i] = y_i -> exp(y_i)
```

## Fields
- `forward::Vector{Function}`: A vector of functions, where `forward[i]` is a function
        `(y_i_::Real, std_i_::Real) -> (y_i::Real, std_i::Real)` that transforms both mean and std of the i-th dimension forward.
- `backward::Vector{Function}`: A vector of functions, where `backward[i]` is a function
        `(y_i::Real) -> y_i_::Real` that transforms mean of the i-th dimension backward.

## Note
Using `SlicedOutputTransform` allows the `TransformedModel` to be sliceable (if the base model is also sliceable),
which enables more efficient parameter estimation.

See also: [`JointOutputTransform`](@ref)
"""
struct SlicedOutputTransform <: OutputTransform
    forward::Vector{Function}
    backward::Vector{Function}

    function SlicedOutputTransform(forward::AbstractVector, backward::AbstractVector)
        @assert length(forward) == length(backward) "Forward and backward function arrays must have the same length."
        return new(collect(Function, forward), collect(Function, backward))
    end
end

function slice(t::SlicedOutputTransform, idx::Int)
    return SlicedOutputTransform([t.forward[idx]], [t.backward[idx]])
end

"""
    TransformedModel(; base_model, input_transform=nothing, output_transform=nothing)

A surrogate model that applies transformations to inputs and/or outputs.

The model first applies the `input_transform` to map `x → x_`, then applies the `base_model`
to get `y_`, and finally applies the `output_transform` to map `y_ → y`.

If both `input_transform` and `output_transform` are `nothing`,
this model behaves identically to the `base_model`.

Both transforms are non-parametric. For a parametrized tarnsformation
of the inputs, which can be trained, consider using a [`ComposedModel`](@ref).

## Keywords
- `base_model::SurrogateModel`: The underlying surrogate model.
- `input_transform::Union{Nothing, InputTransform}`: Optional transformation applied to inputs.
        If `nothing`, no input transformation is applied.
- `output_transform::Union{Nothing, OutputTransform}`: Optional transformation applied to outputs.
        If `nothing`, no output transformation is applied.

See also: [`InputTransform`](@ref), [`OutputTransform`](@ref)
"""
@kwdef struct TransformedModel{
    IN<:Union{Nothing, InputTransform},
    OUT<:Union{Nothing, OutputTransform},
} <: SurrogateModel
    base_model::SurrogateModel
    input_transform::IN = nothing
    output_transform::OUT = nothing
end

"""
    TransformedParams

The parameters of the [`TransformedModel`](@ref).

Simply wraps the parameters of the base model.

## Fields
- `base_params::ModelParams`: The parameters of the base model.
"""
struct TransformedParams <: ModelParams{TransformedModel}
    base_params::ModelParams
end

"""
    TransformedPosterior

The posterior predictive distribution for the [`TransformedModel`](@ref).

## Fields
- `base_posterior::ModelPosterior`: The posterior of the base model.
- `input_transform::Union{Nothing, InputTransform}`: The input transformation.
- `output_transform::Union{Nothing, OutputTransform}`: The output transformation.
"""
struct TransformedPosterior{
    IN<:Union{Nothing, InputTransform},
    OUT<:Union{Nothing, OutputTransform},
} <: ModelPosterior{TransformedModel}
    base_posterior::ModelPosterior
    input_transform::IN
    output_transform::OUT
end


### SurrogateModel API Implementation ###

## Utility Methods

# TODO unimplemented
# function make_discrete(model::TransformedModel, discrete::AbstractVector{Bool}) end

# Model is only sliceable if base model is sliceable AND output transform is SlicedOutputTransform (or nothing)
sliceable(model::TransformedModel{IN, Nothing}) where {IN} = sliceable(model.base_model)
sliceable(model::TransformedModel{IN, SlicedOutputTransform}) where {IN} = sliceable(model.base_model)
sliceable(model::TransformedModel{IN, JointOutputTransform}) where {IN} = false

function slice(model::TransformedModel{IN, Nothing}, idx::Int) where {IN}
    return TransformedModel(
        base_model = slice(model.base_model, idx),
        input_transform = model.input_transform,
        output_transform = nothing,
    )
end
function slice(model::TransformedModel{IN, SlicedOutputTransform}, idx::Int) where {IN}
    return TransformedModel(
        base_model = slice(model.base_model, idx),
        input_transform = model.input_transform,
        output_transform = slice(model.output_transform, idx),
    )
end
function slice(params::TransformedParams, idx::Int)
    return TransformedParams(
        slice(params.base_params, idx)
    )
end

function join_slices(slices::AbstractVector{<:TransformedParams})
    return TransformedParams(
        join_slices(getfield.(slices, :base_params))
    )
end


## Transformations

function model_posterior(model::TransformedModel, params::TransformedParams, data::ExperimentData)
    # Transform the data for the base model
    transformed_data = _transform_data(model, data)
    
    # Get the base posterior
    base_posterior = model_posterior(model.base_model, params.base_params, transformed_data)
    
    return TransformedPosterior(base_posterior, model.input_transform, model.output_transform)
end

# Helper function to transform data
function _transform_data(model::TransformedModel, data::ExperimentData)
    X_ = _transform_input_forward(model.input_transform, data.X)
    Y_ = _transform_output_backward(model.output_transform, data.Y)
    return ExperimentData(X_, Y_)
end

# Transforms: input forward with vectors
function _transform_input_forward(::Nothing, x::AbstractVector{<:Real})
    return x
end
function _transform_input_forward(transform::InputTransform, x::AbstractVector{<:Real})
    return transform.forward(x)
end

# Transforms: input forward with matrices
function _transform_input_forward(::Nothing, X::AbstractMatrix{<:Real})
    return X
end
function _transform_input_forward(transform::InputTransform, X::AbstractMatrix{<:Real})
    return hcat([transform.forward(x) for x in eachcol(X)]...)
end

# Transforms: output backward with matrices
function _transform_output_backward(::Nothing, Y::AbstractMatrix{<:Real})
    return Y
end
function _transform_output_backward(transform::JointOutputTransform, Y::AbstractMatrix{<:Real})
    return hcat([transform.backward(y) for y in eachcol(Y)]...)
end
function _transform_output_backward(transform::SlicedOutputTransform, Y::AbstractMatrix{<:Real})
    Y_model = similar(Y)
    for (i, f) in enumerate(transform.backward)
        Y_model[i, :] = f.(Y[i, :])
    end
    return Y_model
end

# Transforms: output forward with vectors
function _transform_output_forward(::Nothing, μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real})
    return μ, σ
end
function _transform_output_forward(transform::JointOutputTransform, μ_::AbstractVector{<:Real}, σ_::AbstractVector{<:Real})
    return transform.forward(μ_, σ_)
end
function _transform_output_forward(transform::SlicedOutputTransform, μ_::AbstractVector{<:Real}, σ_::AbstractVector{<:Real})
    μ = similar(μ_)
    σ = similar(σ_)
    for i in eachindex(μ_)
        μ[i], σ[i] = transform.forward[i](μ_[i], σ_[i])
    end
    return μ, σ
end

# Transforms: output forward with matrices
function _transform_output_forward(::Nothing, μs::AbstractMatrix{<:Real}, σs::AbstractMatrix{<:Real})
    return μs, σs
end
function _transform_output_forward(transform::JointOutputTransform, μs_::AbstractMatrix{<:Real}, σs_::AbstractMatrix{<:Real})
    μs_out = similar(μs_)
    σs_out = similar(σs_)
    for k in axes(μs_, 1)
        μs_out[k, :], σs_out[k, :] = transform.forward(μs_[k, :], σs_[k, :])
    end
    return μs_out, σs_out
end
function _transform_output_forward(transform::SlicedOutputTransform, μs_::AbstractMatrix{<:Real}, σs_::AbstractMatrix{<:Real})
    μs = similar(μs_)
    σs = similar(σs_)
    n_dims = length(transform.forward)
    for i in 1:n_dims
        for k in axes(μs_, 1)
            μs[k, i], σs[k, i] = transform.forward[i](μs_[k, i], σs_[k, i])
        end
    end
    return μs, σs
end


## Posterior Methods

function mean(post::TransformedPosterior{IN, Nothing}, X::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}}) where {IN}
    X_ = _transform_input_forward(post.input_transform, X)
    return mean(post.base_posterior, X_)
end
function mean(post::TransformedPosterior{IN, OUT}, X::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}}) where {IN, OUT<:OutputTransform}
    μ, _ = mean_and_var(post, X)
    return μ
end

function var(post::TransformedPosterior{IN, Nothing}, X::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}}) where {IN}
    X_ = _transform_input_forward(post.input_transform, X)
    return var(post.base_posterior, X_)
end
function var(post::TransformedPosterior{IN, OUT}, X::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}}) where {IN, OUT<:OutputTransform}
    _, σ2 = mean_and_var(post, X)
    return σ2
end

function cov(post::TransformedPosterior{IN, Nothing}, X::AbstractMatrix{<:Real}) where {IN}
    X_ = _transform_input_forward(post.input_transform, X)
    return cov(post.base_posterior, X_)
end
function cov(post::TransformedPosterior{IN, OUT}, X::AbstractMatrix{<:Real}) where {IN, OUT<:OutputTransform}
    _, Σs = mean_and_cov(post, X)
    return Σs
end

function mean_and_var(post::TransformedPosterior{IN, Nothing}, x::AbstractVector{<:Real}) where {IN}
    x_ = _transform_input_forward(post.input_transform, x)
    return mean_and_var(post.base_posterior, x_)
end
function mean_and_var(post::TransformedPosterior{IN, Nothing}, X::AbstractMatrix{<:Real}) where {IN}
    X_ = _transform_input_forward(post.input_transform, X)
    return mean_and_var(post.base_posterior, X_)
end

function mean_and_var(post::TransformedPosterior{IN, OUT}, x::AbstractVector{<:Real}) where {IN, OUT<:OutputTransform}
    x_ = _transform_input_forward(post.input_transform, x)
    μ_, σ2_ = mean_and_var(post.base_posterior, x_)
    σ_ = sqrt.(σ2_)
    μ, σ = _transform_output_forward(post.output_transform, μ_, σ_)
    return μ, σ .^ 2
end
function mean_and_var(post::TransformedPosterior{IN, OUT}, X::AbstractMatrix{<:Real}) where {IN, OUT<:OutputTransform}
    X_ = _transform_input_forward(post.input_transform, X)
    μs_, σ2s_ = mean_and_var(post.base_posterior, X_)
    σs_ = sqrt.(σ2s_)
    μs, σs = _transform_output_forward(post.output_transform, μs_, σs_)
    return μs, σs .^ 2
end

function mean_and_cov(post::TransformedPosterior{IN, Nothing}, X::AbstractMatrix{<:Real}) where {IN}
    X_ = _transform_input_forward(post.input_transform, X)
    return mean_and_cov(post.base_posterior, X_)
end
function mean_and_cov(post::TransformedPosterior{IN, SlicedOutputTransform}, X::AbstractMatrix{<:Real}) where {IN}
    X_ = _transform_input_forward(post.input_transform, X)
    μs_, Σs_ = mean_and_cov(post.base_posterior, X_)
    
    # For sliced transforms, the covariance is diagonal (each dimension independent)
    μs = similar(μs_)
    for k in axes(Σs_, 3)
        σs_ = sqrt.(diag(Σs_[:, :, k]))
        μs[:, k], σs = _transform_output_forward(post.output_transform, μs_[:, k], σs_)
        for i in axes(Σs_, 1)
            Σs_[i, i, k] = σs[i]^2
        end
    end
    
    return μs, Σs_ # ::Tuple{<:AbstractMatrix{<:Real}, <:AbstractArray{<:Real, 3}}
end
function mean_and_cov(post::TransformedPosterior{IN, JointOutputTransform}, X::AbstractMatrix{<:Real}) where {IN}
    error("Cannot compute covariance with joint output transform. Use mean_and_var instead.")
end


## Parameter Methods

function data_loglike(model::TransformedModel, data::ExperimentData)
    transformed_data = _transform_data(model, data)
    ll_base = data_loglike(model.base_model, transformed_data)
    
    function ll_data(params::TransformedParams)
        return ll_base(params.base_params)
    end
end

function params_loglike(model::TransformedModel)
    ll_base = params_loglike(model.base_model)
    
    function ll_params(params::TransformedParams)
        return ll_base(params.base_params)
    end
end

function _params_sampler(model::TransformedModel)
    base_sampler = _params_sampler(model.base_model)
    
    function sample(rng::AbstractRNG)
        base_params = base_sampler(rng)
        return TransformedParams(base_params)
    end
end

function vectorizer(model::TransformedModel)
    base_vec, base_devec = vectorizer(model.base_model)
    
    function vectorize(params::TransformedParams)
        return base_vec(params.base_params)
    end
    
    function devectorize(params::TransformedParams, ps::AbstractVector{<:Real})
        base_params = base_devec(params.base_params, ps)
        return TransformedParams(base_params)
    end

    return vectorize, devectorize
end

function bijector(model::TransformedModel)
    return bijector(model.base_model)
end
