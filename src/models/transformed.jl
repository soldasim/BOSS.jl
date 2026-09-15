
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

`forward` must implement two methods (via ordinary Julia multiple dispatch on the same function):
a pointwise map `(y_) -> y`, and a mean/std propagation map `(y_, std_) -> (y, std)`. `backward`
maps user-space `y` back to model-space `y_`.

The pointwise method is used to transform individual predictive samples/quadrature atoms (see
[`predictive_samples`](@ref)) exactly — this is the only way `predictive_samples` can be supported
through an output transform. The mean/std method is used for the `GaussianPredictive` case, where
there are no discrete atoms, only a `(mean, std)` pair to propagate; for a nonlinear transform this
should propagate uncertainty appropriately, typically the delta-method linearization of the
pointwise map using its derivative. For example, for `y = log(y_)`:
```julia
forward(y_) = log.(y_)
forward(y_, std_) = log.(y_), std_ ./ y_  # derivative: dy/dy_ = 1/y_
backward(y) = exp.(y)
```

## Fields
- `forward::Function`: A function implementing both
        `(y_::AbstractVector{<:Real}) -> y::AbstractVector{<:Real}` and
        `(y_::AbstractVector{<:Real}, std_::AbstractVector{<:Real}) -> (y::AbstractVector{<:Real}, std::AbstractVector{<:Real})`.
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

Each `forward[i]` must implement two methods (via ordinary Julia multiple dispatch on the same
function): a pointwise map `(y_i_) -> y_i`, and a mean/std propagation map
`(y_i_, std_i_) -> (y_i, std_i)`. Each `backward[i]` maps user-space `y_i` back to model-space `y_i_`.

The pointwise method is used to transform individual predictive samples/quadrature atoms of this
dimension (see [`predictive_samples`](@ref)) exactly — this is the only way `predictive_samples`
can be supported through an output transform. The mean/std method is used for the
`GaussianPredictive` case, where there are no discrete atoms, only a `(mean, std)` pair to
propagate; for a nonlinear transform this should propagate uncertainty appropriately, typically
the delta-method linearization of the pointwise map using its derivative. For example, for
`y_i = log(y_i_)`:
```julia
forward[i](y_i_) = log(y_i_)
forward[i](y_i_, std_i_) = log(y_i_), std_i_ / y_i_  # derivative: dy_i/dy_i_ = 1/y_i_
backward[i](y_i) = exp(y_i)
```

## Fields
- `forward::Vector{Function}`: A vector of functions, where `forward[i]` implements both
        `(y_i_::Real) -> y_i::Real` and `(y_i_::Real, std_i_::Real) -> (y_i::Real, std_i::Real)`
        for the i-th dimension.
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
of the inputs, which can be trained, consider using a `ComposedModel`.

## Keywords
- `base_model::SurrogateModel`: The underlying surrogate model.
- `input_transform::Union{Nothing, InputTransform}`: Optional transformation applied to inputs.
        If `nothing`, no input transformation is applied.
- `output_transform::Union{Nothing, OutputTransform}`: Optional transformation applied to outputs.
        If `nothing`, no output transformation is applied.

## Predictive Kind

`base_model`'s type is tracked as a type parameter (`B`, listed first) so that
[`predictive_kind`](@ref) can dispatch straight to it, for any `output_transform`. When
`output_transform === nothing`, the output-space predictive is identical to `base_model`'s own.
When an `OutputTransform` is present, its required pointwise `forward(y_) -> y` method (see
[`OutputTransform`](@ref)) lets [`predictive_samples`](@ref) push each of `base_model`'s own
sample atoms through the transform exactly, so a `SampledPredictive` base model (e.g.
[`WarpedGaussianProcess`](@ref)) stays `SampledPredictive` through the wrapper either way.

See also: [`InputTransform`](@ref), [`OutputTransform`](@ref)
"""
@kwdef struct TransformedModel{
    B<:SurrogateModel,
    IN<:Union{Nothing, InputTransform},
    OUT<:Union{Nothing, OutputTransform},
} <: SurrogateModel
    base_model::B
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
    B<:SurrogateModel,
    IN<:Union{Nothing, InputTransform},
    OUT<:Union{Nothing, OutputTransform},
} <: ModelPosterior{TransformedModel{B, IN, OUT}}
    base_posterior::ModelPosterior{B}
    input_transform::IN
    output_transform::OUT
end

"""
    TransformedPosteriorSlice

The posterior predictive distribution for a single output dimension of a [`TransformedModel`](@ref).

Wraps a `ModelPosteriorSlice{B}` of the base model directly, rather than being obtained generically
by slicing a joint [`TransformedPosterior`](@ref) — going through the joint posterior would require
first bundling the base model's per-dimension slices into a `ModelPosterior`, which is unavailable
for `predictive_samples` when the base model only samples independently per dimension (e.g.
[`WarpedGaussianProcess`](@ref); see [`predictive_samples`](@ref)).

## Fields
- `base_posterior_slice::ModelPosteriorSlice`: The posterior slice of the base model.
- `input_transform::Union{Nothing, InputTransform}`: The input transformation.
- `output_transform::Union{Nothing, OutputTransform}`: The (per-dimension) output transformation.
"""
struct TransformedPosteriorSlice{
    B<:SurrogateModel,
    IN<:Union{Nothing, InputTransform},
    OUT<:Union{Nothing, OutputTransform},
} <: ModelPosteriorSlice{TransformedModel{B, IN, OUT}}
    base_posterior_slice::ModelPosteriorSlice{B}
    input_transform::IN
    output_transform::OUT
end


### SurrogateModel API Implementation ###

## Utility Methods

# TODO unimplemented
# function make_discrete(model::TransformedModel, discrete::AbstractVector{Bool}) end

# Model is only sliceable if base model is sliceable AND output transform is SlicedOutputTransform (or nothing)
sliceable(::Type{<:TransformedModel{B, IN, OUT}}) where {B, IN, OUT<:Union{Nothing, SlicedOutputTransform}} = sliceable(B)
sliceable(::Type{<:TransformedModel{B, IN, JointOutputTransform}}) where {B, IN} = false

# Model is only dim-independent (given parameters) if base model is sliceable AND output transform is SlicedOutputTransform (or nothing)
dimension_independent_given_parameters(::Type{<:TransformedModel{B, IN, OUT}}) where {B, IN, OUT<:Union{Nothing, SlicedOutputTransform}} = dimension_independent_given_parameters(B)
dimension_independent_given_parameters(::Type{<:TransformedModel{B, IN, JointOutputTransform}}) where {B, IN} = false

function slice(model::TransformedModel{B, IN, Nothing}, idx::Int) where {B, IN}
    return TransformedModel(
        base_model = slice(model.base_model, idx),
        input_transform = model.input_transform,
        output_transform = nothing,
    )
end
function slice(model::TransformedModel{B, IN, SlicedOutputTransform}, idx::Int) where {B, IN}
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

# See `TransformedModel`'s "Predictive Kind" section above.
predictive_kind(::Type{<:TransformedModel{B, IN, OUT}}) where {B, IN, OUT} = predictive_kind(B)


## Transformations

function model_posterior(model::TransformedModel, params::TransformedParams, data::ExperimentData)
    # Transform the data for the base model
    transformed_data = _transform_data(model, data)

    # Get the base posterior
    base_posterior = model_posterior(model.base_model, params.base_params, transformed_data)

    return TransformedPosterior(base_posterior, model.input_transform, model.output_transform)
end

# `idx`, not `slice`, to avoid shadowing the `slice(::SurrogateModel, ::Int)` function called below.
function model_posterior_slice(model::TransformedModel{B, IN, Nothing}, params::TransformedParams, data::ExperimentData, idx::Int) where {B, IN}
    transformed_data = _transform_data(model, data)
    base_posterior_slice = model_posterior_slice(model.base_model, params.base_params, transformed_data, idx)
    return TransformedPosteriorSlice(base_posterior_slice, model.input_transform, model.output_transform)
end
function model_posterior_slice(model::TransformedModel{B, IN, SlicedOutputTransform}, params::TransformedParams, data::ExperimentData, idx::Int) where {B, IN}
    transformed_data = _transform_data(model, data)
    base_posterior_slice = model_posterior_slice(model.base_model, params.base_params, transformed_data, idx)
    return TransformedPosteriorSlice(base_posterior_slice, model.input_transform, slice(model.output_transform, idx))
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
    for k in axes(μs_, 2)  # iterate over columns (points)
        μs_out[:, k], σs_out[:, k] = transform.forward(μs_[:, k], σs_[:, k])
    end
    return μs_out, σs_out
end
function _transform_output_forward(transform::SlicedOutputTransform, μs_::AbstractMatrix{<:Real}, σs_::AbstractMatrix{<:Real})
    μs = similar(μs_)
    σs = similar(σs_)
    n_dims = length(transform.forward)
    for i in 1:n_dims  # iterate over output dimensions (rows)
        for k in axes(μs_, 2)  # iterate over points (columns)
            μs[i, k], σs[i, k] = transform.forward[i](μs_[i, k], σs_[i, k])
        end
    end
    return μs, σs
end

# Transforms: pointwise output forward
function _transform_output_forward_samples(transform::JointOutputTransform, y_::AbstractVector{<:Real})
    return transform.forward(y_)
end
function _transform_output_forward_samples(transform::SlicedOutputTransform, y_::AbstractVector{<:Real})
    return [f(v) for (f, v) in zip(transform.forward, y_)]
end
function _transform_output_forward_samples(transform::OutputTransform, Ys_::AbstractMatrix{<:Real})
    return hcat([_transform_output_forward_samples(transform, Ys_[:, k]) for k in axes(Ys_, 2)]...)
end
function _transform_output_forward_samples(transform::OutputTransform, Ys_::AbstractArray{<:Real, 3})
    return cat([_transform_output_forward_samples(transform, Ys_[:, :, j]) for j in axes(Ys_, 3)]...; dims=3)
end

# Weighted mean/var/cov copmutations
function _weighted_mean_and_var(ys::AbstractVector{<:Real}, ws::AbstractVector{<:Real})
    μ = sum(ws .* ys)
    σ2 = sum(ws .* (ys .- μ) .^ 2)
    return μ, σ2
end
function _weighted_mean_and_var(Ys::AbstractMatrix{<:Real}, Ws::AbstractMatrix{<:Real})
    μs = vec(sum(Ws .* Ys; dims=1))
    σ2s = vec(sum(Ws .* (Ys .- μs') .^ 2; dims=1))
    return μs, σ2s
end

function _weighted_mean_and_var_joint(Ys::AbstractMatrix{<:Real}, Ws::AbstractMatrix{<:Real})
    @assert size(Ws, 1) == 1
    ws = vec(Ws)
    μ = Ys * ws
    σ2 = (Ys .- μ) .^ 2 * ws
    return μ, σ2
end
function _weighted_mean_and_var_joint(Ys::AbstractArray{<:Real, 3}, Ws::AbstractArray{<:Real, 3})
    @assert size(Ws, 1) == 1
    y_dim, K, n = size(Ys)
    μs = Matrix{eltype(Ys)}(undef, y_dim, n)
    σ2s = Matrix{eltype(Ys)}(undef, y_dim, n)
    for j in 1:n
        μs[:, j], σ2s[:, j] = _weighted_mean_and_var_joint(Ys[:, :, j], Ws[:, :, j])
    end
    return μs, σ2s
end

function _weighted_mean_and_cov_joint(Ys::AbstractMatrix{<:Real}, Ws::AbstractMatrix{<:Real})
    @assert size(Ws, 1) == 1
    ws = vec(Ws)
    μ = Ys * ws
    centered = Ys .- μ
    Σ = (centered .* ws') * centered'
    return μ, Σ
end
function _weighted_mean_and_cov_joint(Ys::AbstractArray{<:Real, 3}, Ws::AbstractArray{<:Real, 3})
    @assert size(Ws, 1) == 1
    y_dim, K, n = size(Ys)
    μs = Matrix{eltype(Ys)}(undef, y_dim, n)
    Σs = Array{eltype(Ys), 3}(undef, y_dim, y_dim, n)
    for j in 1:n
        μs[:, j], Σs[:, :, j] = _weighted_mean_and_cov_joint(Ys[:, :, j], Ws[:, :, j])
    end
    return μs, Σs
end


## Joint Posterior Methods

function mean(post::TransformedPosterior{B, IN, Nothing}, X::AbstractVecOrMat{<:Real}) where {B, IN}
    X_ = _transform_input_forward(post.input_transform, X)
    return mean(post.base_posterior, X_)
end
function mean(post::TransformedPosterior{B, IN, OUT}, X::AbstractVecOrMat{<:Real}) where {B, IN, OUT<:OutputTransform}
    μ, _ = mean_and_var(post, X)
    return μ
end

function var(post::TransformedPosterior{B, IN, Nothing}, X::AbstractVecOrMat{<:Real}) where {B, IN}
    X_ = _transform_input_forward(post.input_transform, X)
    return var(post.base_posterior, X_)
end
function var(post::TransformedPosterior{B, IN, OUT}, X::AbstractVecOrMat{<:Real}) where {B, IN, OUT<:OutputTransform}
    _, σ2 = mean_and_var(post, X)
    return σ2
end

function cov(post::TransformedPosterior{B, IN, Nothing}, X::AbstractMatrix{<:Real}) where {B, IN}
    X_ = _transform_input_forward(post.input_transform, X)
    return cov(post.base_posterior, X_)
end
function cov(post::TransformedPosterior{B, IN, OUT}, X::AbstractMatrix{<:Real}) where {B, IN, OUT<:OutputTransform}
    _, Σs = mean_and_cov(post, X)
    return Σs
end

function mean_and_var(post::TransformedPosterior{B, IN, Nothing}, x::AbstractVecOrMat{<:Real}) where {B, IN}
    x_ = _transform_input_forward(post.input_transform, x)
    return mean_and_var(post.base_posterior, x_)
end
function mean_and_var(post::TransformedPosterior{B, IN, OUT}, x::AbstractVecOrMat{<:Real}) where {B, IN, OUT<:OutputTransform}
    return _mean_and_var_kind(predictive_kind(B), post, x)
end

function _mean_and_var_kind(::SampledPredictive, post::TransformedPosterior, x::AbstractVecOrMat{<:Real})
    Ys, Ws = predictive_samples(post, x)
    return _weighted_mean_and_var_joint(Ys, Ws)
end
function _mean_and_var_kind(::GaussianPredictive, post::TransformedPosterior, x::AbstractVecOrMat{<:Real})
    x_ = _transform_input_forward(post.input_transform, x)
    μ_, σ2_ = mean_and_var(post.base_posterior, x_)
    σ_ = sqrt.(σ2_)
    μ, σ = _transform_output_forward(post.output_transform, μ_, σ_)
    return μ, σ .^ 2
end

function mean_and_cov(post::TransformedPosterior{B, IN, Nothing}, X::AbstractMatrix{<:Real}) where {B, IN}
    X_ = _transform_input_forward(post.input_transform, X)
    return mean_and_cov(post.base_posterior, X_)
end
function mean_and_cov(post::TransformedPosterior{B, IN, OUT}, X::AbstractMatrix{<:Real}) where {B, IN, OUT<:OutputTransform}
    return _mean_and_cov_kind(predictive_kind(B), post, X)
end

function _mean_and_cov_kind(::SampledPredictive, post::TransformedPosterior, X::AbstractMatrix{<:Real})
    Ys, Ws = predictive_samples(post, X)
    return _weighted_mean_and_cov_joint(Ys, Ws)
end
function _mean_and_cov_kind(::GaussianPredictive, post::TransformedPosterior{B, IN, SlicedOutputTransform}, X::AbstractMatrix{<:Real}) where {B, IN}
    X_ = _transform_input_forward(post.input_transform, X)
    μs_, Σs_ = mean_and_cov(post.base_posterior, X_)

    # For sliced transforms, the covariance is diagonal (each dimension independent)
    # μs_ has rows=output_dims, columns=points
    # Σs_[:,:,k] is covariance matrix for point k
    μs = similar(μs_)
    n_points = size(μs_, 2)
    for k in 1:n_points  # iterate over points
        σs_ = sqrt.(diag(Σs_[:, :, k]))
        μs[:, k], σs = _transform_output_forward(post.output_transform, μs_[:, k], σs_)
        for i in axes(Σs_, 1)  # iterate over output dimensions
            Σs_[i, i, k] = σs[i]^2
        end
    end

    return μs, Σs_ # ::Tuple{<:AbstractMatrix{<:Real}, <:AbstractArray{<:Real, 3}}
end
function _mean_and_cov_kind(::GaussianPredictive, post::TransformedPosterior{B, IN, JointOutputTransform}, X::AbstractMatrix{<:Real}) where {B, IN}
    error("Cannot compute covariance with joint output transform. Use mean_and_var instead.")
end

function predictive_samples(post::TransformedPosterior{B, IN, Nothing}, x::AbstractVecOrMat{<:Real}; kwargs...) where {B, IN}
    x_ = _transform_input_forward(post.input_transform, x)
    return predictive_samples(post.base_posterior, x_; kwargs...)
end
function predictive_samples(post::TransformedPosterior{B, IN, OUT}, x::AbstractVecOrMat{<:Real}; lower::Real=-Inf, upper::Real=Inf, kwargs...) where {B, IN, OUT<:OutputTransform}
    (isinf(lower) && isinf(upper)) || error(
        "`predictive_samples` with finite `lower`/`upper` is not supported for a joint " *
        "`TransformedPosterior{$B, $IN, $OUT}` -- the bounds would need to be mapped through " *
        "`output_transform`'s backward direction first, which isn't implemented for this case."
    )
    x_ = _transform_input_forward(post.input_transform, x)
    Ys_, Ws = predictive_samples(post.base_posterior, x_; kwargs...)
    Ys = _transform_output_forward_samples(post.output_transform, Ys_)
    return Ys, Ws
end


## Sliced Posterior Methods

function mean(post::TransformedPosteriorSlice{B, IN, Nothing}, X::AbstractVecOrMat{<:Real}) where {B, IN}
    X_ = _transform_input_forward(post.input_transform, X)
    return mean(post.base_posterior_slice, X_)
end
function mean(post::TransformedPosteriorSlice{B, IN, OUT}, X::AbstractVecOrMat{<:Real}) where {B, IN, OUT<:OutputTransform}
    μ, _ = mean_and_var(post, X)
    return μ
end

function var(post::TransformedPosteriorSlice{B, IN, Nothing}, X::AbstractVecOrMat{<:Real}) where {B, IN}
    X_ = _transform_input_forward(post.input_transform, X)
    return var(post.base_posterior_slice, X_)
end
function var(post::TransformedPosteriorSlice{B, IN, OUT}, X::AbstractVecOrMat{<:Real}) where {B, IN, OUT<:OutputTransform}
    _, σ2 = mean_and_var(post, X)
    return σ2
end

function cov(post::TransformedPosteriorSlice{B, IN, Nothing}, X::AbstractMatrix{<:Real}) where {B, IN}
    X_ = _transform_input_forward(post.input_transform, X)
    return cov(post.base_posterior_slice, X_)
end
function cov(post::TransformedPosteriorSlice{B, IN, OUT}, X::AbstractMatrix{<:Real}) where {B, IN, OUT<:OutputTransform}
    _, Σs = mean_and_cov(post, X)
    return Σs
end

function mean_and_var(post::TransformedPosteriorSlice{B, IN, Nothing}, x::AbstractVecOrMat{<:Real}) where {B, IN}
    x_ = _transform_input_forward(post.input_transform, x)
    return mean_and_var(post.base_posterior_slice, x_)
end

function mean_and_var(post::TransformedPosteriorSlice{B, IN, OUT}, x::AbstractVecOrMat{<:Real}) where {B, IN, OUT<:OutputTransform}
    return _mean_and_var_kind(predictive_kind(B), post, x)
end

function _mean_and_var_kind(::SampledPredictive, post::TransformedPosteriorSlice, x::AbstractVecOrMat{<:Real})
    ys, ws = predictive_samples(post, x)
    return _weighted_mean_and_var(ys, ws)
end
function _mean_and_var_kind(::GaussianPredictive, post::TransformedPosteriorSlice, x::AbstractVector{<:Real})
    x_ = _transform_input_forward(post.input_transform, x)
    μ_, σ2_ = mean_and_var(post.base_posterior_slice, x_) # scalars
    σ_ = sqrt(σ2_)
    # wrap the scalar mean/std as length-1 vectors to reuse `_transform_output_forward`, then unwrap.
    μ, σ = _transform_output_forward(post.output_transform, [μ_], [σ_])
    return μ[1], σ[1]^2
end
function _mean_and_var_kind(::GaussianPredictive, post::TransformedPosteriorSlice, X::AbstractMatrix{<:Real})
    X_ = _transform_input_forward(post.input_transform, X)
    μs_, σ2s_ = mean_and_var(post.base_posterior_slice, X_) # vectors, length n_points
    σs_ = sqrt.(σ2s_)
    # reshape to (1, n_points) to reuse the matrix-form `_transform_output_forward` method, then unwrap.
    μs, σs = _transform_output_forward(post.output_transform, reshape(μs_, 1, :), reshape(σs_, 1, :))
    return vec(μs), vec(σs) .^ 2
end

function mean_and_cov(post::TransformedPosteriorSlice{B, IN, Nothing}, X::AbstractMatrix{<:Real}) where {B, IN}
    X_ = _transform_input_forward(post.input_transform, X)
    return mean_and_cov(post.base_posterior_slice, X_)
end
function mean_and_cov(post::TransformedPosteriorSlice{B, IN, SlicedOutputTransform}, X::AbstractMatrix{<:Real}) where {B, IN}
    return _mean_and_cov_kind(predictive_kind(B), post, X)
end

function _mean_and_cov_kind(::SampledPredictive, post::TransformedPosteriorSlice, X::AbstractMatrix{<:Real})
    X_ = _transform_input_forward(post.input_transform, X)
    _, Σs_ = mean_and_cov(post.base_posterior_slice, X_) # Σs_: (n_points, n_points), model-space

    # Off-diagonal (cross-point) covariance can't be recovered from `predictive_samples` (each
    # point's atoms are that point's own marginal predictive, not jointly sampled across points),
    # so it stays in model-space scale, as in the `GaussianPredictive` method below; only the mean
    # and the diagonal (per-point variance) are corrected, now exactly via the atoms instead of the
    # delta-method linearization used there.
    Ys, Ws = predictive_samples(post, X)
    μs, σ2s = _weighted_mean_and_var(Ys, Ws)
    for i in axes(Σs_, 1)
        Σs_[i, i] = σ2s[i]
    end
    return μs, Σs_
end
function _mean_and_cov_kind(::GaussianPredictive, post::TransformedPosteriorSlice, X::AbstractMatrix{<:Real})
    X_ = _transform_input_forward(post.input_transform, X)
    μs_, Σs_ = mean_and_cov(post.base_posterior_slice, X_) # μs_: (n_points,), Σs_: (n_points, n_points), model-space

    # As in the joint `SlicedOutputTransform` `mean_and_cov` above: only the diagonal (per-point
    # variance) is corrected via the output transform; off-diagonal (cross-point) covariance is
    # left in the model-space scale.
    σs_ = sqrt.(diag(Σs_))
    μs, σs = _transform_output_forward(post.output_transform, reshape(μs_, 1, :), reshape(σs_, 1, :))
    for i in axes(Σs_, 1)
        Σs_[i, i] = σs[1, i]^2
    end

    return vec(μs), Σs_ # ::Tuple{<:AbstractVector{<:Real}, <:AbstractMatrix{<:Real}}
end

function predictive_samples(post::TransformedPosteriorSlice{B, IN, Nothing}, x::AbstractVecOrMat{<:Real}; kwargs...) where {B, IN}
    x_ = _transform_input_forward(post.input_transform, x)
    return predictive_samples(post.base_posterior_slice, x_; kwargs...)
end
function predictive_samples(post::TransformedPosteriorSlice{B, IN, SlicedOutputTransform}, x::AbstractVecOrMat{<:Real}; lower::Real=-Inf, upper::Real=Inf, kwargs...) where {B, IN}
    x_ = _transform_input_forward(post.input_transform, x)
    if isinf(lower) && isinf(upper)
        lower_, upper_ = lower, upper
    else
        bwd = post.output_transform.backward[1]
        lo_ = isinf(lower) ? lower : bwd(lower)
        hi_ = isinf(upper) ? upper : bwd(upper)
        lower_, upper_ = min(lo_, hi_), max(lo_, hi_)
    end
    ys_, ws = predictive_samples(post.base_posterior_slice, x_; lower=lower_, upper=upper_, kwargs...)
    ys = post.output_transform.forward[1].(ys_)
    return ys, ws
end


## Parameter Methods

function data_loglike(model::TransformedModel, data::ExperimentData)
    transformed_data = _transform_data(model, data)
    ll_base = data_loglike(model.base_model, transformed_data)

    function ll_data(params::TransformedParams)
        return ll_base(params.base_params)
    end
end

function params_logprior(model::TransformedModel)
    ll_base = params_logprior(model.base_model)

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
