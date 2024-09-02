
"""
Update model parameters.
"""
function update_parameters!(::Type{T}, data::ExperimentDataPrior, params::ModelParams) where {T<:ModelFit}
    return ExperimentDataMAP(
        data.X,
        data.Y,
        params,
        true, # consistent
    )
end
function update_parameters!(::Type{T}, data::ExperimentDataPrior, params::AbstractVector{<:ModelParams}) where {T<:ModelFit}
    return ExperimentDataBI(
        data.X,
        data.Y,
        params,
        true, # consistent
    )
end
function update_parameters!(::Type{T}, data::ExperimentDataPost{T}, params) where {T<:ModelFit}
    data.params = params
    data.consistent = true
    return data
end
function update_parameters!(::Type{T}, data::ExperimentDataPost, params) where {T<:ModelFit}
    throw(ArgumentError("Inconsistent experiment data!\nCannot switch from MAP to BI or vice-versa."))
end

"""
    augment_dataset!(data::ExperimentDataPost, x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    augment_dataset!(data::ExperimentDataPost, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})

Add one (as vectors) or more (as matrices) datapoints to the dataset.
"""
function augment_dataset!(data::ExperimentDataPost, X::AbstractArray{<:Real}, Y::AbstractArray{<:Real})
    data.X = hcat(data.X, X)
    data.Y = hcat(data.Y, Y)
    data.consistent = false
    return data
end

"""
    consistent(::ExperimentData)

Returns true if the parameters in the data are estimated
using to the current dataset (X, Y).

Returns false if the dataset have been augmented
since the parameters have been estimated.
"""
consistent(data::ExperimentDataPrior) = false
consistent(data::ExperimentDataPost) = data.consistent

"""
    eachsample(::ExperimentData)

Return a `BISamples` iterator over the hyperparameter samples contained in the data.
"""
eachsample(data::ExperimentDataBI) = BISamples(data)

"""
    sample_count(::ExperimentDataBI)

Return the number of hyperparameter samples stored in the data.
"""
sample_count(data::ExperimentDataBI) = length(data.params)

"""
    get_sample(::ExperimentData, ::Int)

Return a single hyperparameter sample as an instance of `ExperimentDataMAP`.
"""
function get_sample(data::ExperimentDataBI, idx::Int)
    return ExperimentDataMAP(
        data.X,
        data.Y,
        data.params[idx],
        data.consistent,
    )
end

"""
Iterator over BI samples contained in `ExperimentDataBI` structure.
The returned elements are instances of `ExperimentDataMAP`.
"""
struct BISamples
    data::ExperimentDataBI
end

Base.getindex(samples::BISamples, idx::Int) = get_sample(samples.data, idx)
Base.length(samples::BISamples) = sample_count(samples.data)
Base.eltype(::BISamples) = ExperimentDataMAP

function Base.iterate(samples::BISamples)
    if length(samples) == 0
        return nothing
    else
        return samples[1], 1
    end
end
function Base.iterate(samples::BISamples, state::Int)
    state += 1
    if state > length(samples)
        return nothing
    else
        return samples[state], state
    end
end
