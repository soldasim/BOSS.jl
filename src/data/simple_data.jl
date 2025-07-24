
"""
    SimpleData(X, Y)

Stores all the data collected during the optimization. Performs no additional preprocessing.

## Fields
- `X::AbstractMatrix{<:Real}`: Contains the objective function inputs as columns.
- `Y::AbstractMatrix{<:Real}`: Contains the objective function outputs as columns.
"""
@kwdef struct SimpleData{
    XT<:AbstractMatrix{<:Real},
    YT<:AbstractMatrix{<:Real},
} <: ExperimentData
    X::XT
    Y::YT

    function SimpleData(X::XT, Y::YT) where {XT, YT}
        @assert size(X, 2) == size(Y, 2)
        return new{XT, YT}(X, Y)
    end
end

function augment_dataset(data::SimpleData, X::AbstractArray{<:Real}, Y::AbstractArray{<:Real})
    return SimpleData(
        hcat(data.X, X),
        hcat(data.Y, Y),
    )
end

function update_dataset(data::SimpleData, X::AbstractArray{<:Real}, Y::AbstractArray{<:Real})
    return SimpleData(
        X,
        Y,
    )
end

function slice(data::SimpleData, idx::Int)
    return SimpleData(
        data.X,
        data.Y[idx:idx,:],
    )
end
