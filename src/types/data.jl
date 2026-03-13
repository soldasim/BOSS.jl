
"""
    ExperimentData

An abstract type for different types storing (and possibly preprocessing) the experiment data.

## Interface

All subtypes of `ExperimentData` should contain the following fields:
- `X::AbstractMatrix{<:Real}`: The input data matrix.
- `Y::AbstractMatrix{<:Real}`: The (possibly pre-processed) output data matrix.

All subtypes of `ExperimentData` *should* implement the following methods:
- `augment_dataset(::ExperimentData, x, out) -> ::ExperimentData`
- `slice(::ExperimentData, idx::Int) -> ::ExperimentData`
"""
abstract type ExperimentData end

x_dim(data::ExperimentData) = size(data.X, 1)
y_dim(data::ExperimentData) = size(data.Y, 1)

Base.length(data::ExperimentData) = size(data.X, 2)
Base.isempty(data::ExperimentData) = isempty(data.X)

"""
    augment_dataset(::ExperimentData, x, out) -> ::ExperimentData

Return a new `ExperimentData` instance containing the current dataset
augmented by the provided data new data point `x` and outputs `out`.
"""
function augment_dataset end

# docstring in `src/types/problem.jl`
# function slice end
