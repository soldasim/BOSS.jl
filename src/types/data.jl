
"""
    ExperimentData(X, Y)

Stores all the data collected during the optimization.

At least one initial datapoint has to be provided (purely for implementation reasons).
One can for example use LatinHypercubeSampling.jl to obtain a small intial grid,
or provide a single random initial datapoint.

# Keywords
- `X::AbstractMatrix{<:Real}`: Contains the objective function inputs as columns.
- `Y::AbstractMatrix{<:Real}`: Contains the objective function outputs as columns.
"""
@kwdef struct ExperimentData{
    T<:AbstractMatrix{<:Real},
}
    X::T
    Y::T

    function ExperimentData(X::T, Y::T) where {T}
        @assert size(X, 2) == size(Y, 2)
        return new{T}(X, Y)
    end
end

x_dim(d::ExperimentData) = size(d.X, 1)
y_dim(d::ExperimentData) = size(d.Y, 1)

Base.length(data::ExperimentData) = size(data.X, 2)
Base.isempty(data::ExperimentData) = isempty(data.X)

function augment_dataset(data::ExperimentData, X::AbstractArray{<:Real}, Y::AbstractArray{<:Real})
    return ExperimentData(
        hcat(data.X, X),
        hcat(data.Y, Y),
    )
end

function slice(data::ExperimentData, idx::Int)
    return ExperimentData(
        data.X,
        data.Y[idx:idx,:],
    )
end
