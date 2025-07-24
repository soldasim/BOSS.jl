
"""
    ExperimentData

An abstract type for different types storing (and possibly preprocessing) the experiment data.

## Interface

All subtypes of `ExperimentData` should contain the following fields:
- `X::AbstractMatrix{<:Real}`: The input data matrix.
- `Y::AbstractMatrix{<:Real}`: The (possibly pre-processed) output data matrix.

All subtypes of `ExperimentData` *should* implement the following methods:
- `augment_dataset(::ExperimentData, X, Y) -> ::ExperimentData`: Returns a new `ExperimentData` instance
        containing the current dataset *augmented by* the provided data `X`, `Y`.
- `update_dataset(::ExperimentData, X, Y) -> ::ExperimentData`: Returns a new `ExperimentData` instance
        containing *only* the new provided data `X`, `Y`.
- `slice(::ExperimentData, idx::Int) -> ::ExperimentData`: Returns a new `ExperimentData` instance
        containing only the output dimension specified by the `idx` index.
"""
abstract type ExperimentData end

x_dim(data::ExperimentData) = size(data.X, 1)
y_dim(data::ExperimentData) = size(data.Y, 1)

Base.length(data::ExperimentData) = size(data.X, 2)
Base.isempty(data::ExperimentData) = isempty(data.X)

"""
    augment_dataset(::ExperimentData, X, Y) -> ::ExperimentData

Return a new `ExperimentData` instance containing the old dataset *augmented by* the provided data `X`, `Y`.

See also: `update_dataset`
"""
function augment_dataset end

"""
    update_dataset(::ExperimentData, X, Y) -> ::ExperimentData

Return a new `ExperimentData` instance containing *only* the new provided data `X`, `Y`.

See also: `augment_dataset`
"""
function update_dataset end

# docstring in `src/types/problem.jl`
# function slice end
