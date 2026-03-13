
"""
    GradientData(X, Y, dY)

Stores experiment data along with gradient observations for use with
[`GradientGaussianProcess`](@ref).

## Fields

- `X::AbstractMatrix{<:Real}`: Input data, shape `x_dim × n`.
- `Y::AbstractMatrix{<:Real}`: Function value data, shape `y_dim × n`.
- `dY::AbstractArray{<:Real, 3}`: Jacobian data.
  - When unsliced: shape `y_dim × x_dim × n`. Each `dY[:, :, j]` is the `y_dim × x_dim` Jacobian matrix at `X[:, j]`.
  - When sliced to single output: shape `x_dim × n`. Gradient of single output for each sample.
  Element `dY[i, k, j]` represents `∂yᵢ/∂xₖ` where `yᵢ` is the i-th output and `xₖ` is the k-th input.

The simulator `f(x)` should return `(y, ∇y)` where `∇y = vec(J')` and
`J = ForwardDiff.jacobian(f_y, x)` is the `y_dim × x_dim` Jacobian.
This will be reshaped into the 3D array format.

## See Also

[`GradientGaussianProcess`](@ref)
"""
struct GradientData{
    XT<:AbstractMatrix{<:Real},
    YT<:AbstractMatrix{<:Real},
    DYT<:AbstractArray{<:Real, 3},
} <: ExperimentData
    X::XT
    Y::YT
    dY::DYT

    function GradientData(X::XT, Y::YT, dY::DYT) where {XT, YT, DYT}
        @assert size(X, 2) == size(Y, 2) == size(dY, 3)
        return new{XT, YT, DYT}(X, Y, dY)
    end
end

"""
    augment_dataset(::GradientData, x, results)
    augment_dataset(::GradientData, x, y, J)

Appends the new function value `y` and Jacobian `J` to the dataset.
The Jacobian `J` should be a `y_dim × x_dim` matrix.
"""
function augment_dataset(
    data::GradientData,
    x::AbstractVector{<:Real},
    results::Tuple,
)
    y, J = results
    return augment_dataset(data, x, y, J)
end
function augment_dataset(
    data::GradientData,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    J::AbstractMatrix{<:Real},
)
    return GradientData(
        hcat(data.X, x),
        hcat(data.Y, y),
        cat(data.dY, J; dims=3),
    )
end

"""
    slice(::GradientData, idx)

Extract data for output dimension `idx`.
After slicing, `data.dY` has shape `x_dim × n` and contains the gradient of output `idx`.
"""
function slice(data::GradientData, idx::Int)
    return GradientData(
        data.X,
        data.Y[idx:idx, :],
        data.dY[idx:idx, :, :],  # Keep as 3D for consistency, even if single output
    )
end
