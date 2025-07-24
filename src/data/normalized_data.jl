
"""
    NormalizedData(X, Y_orig; y_lb, y_ub)
    NormalizedData(::SimpleData; y_lb, y_ub)

Stores all the data collected during the optimization.

Normalizes the output data according to the provided `y_lb`, `y_ub` vectors.

## Fields
- `X::AbstractMatrix{<:Real}`: The input data matrix.
- `Y::AbstractMatrix{<:Real}`: The normalized output data matrix.
- `Y_orig::AbstractMatrix{<:Real}`: The original output data matrix containing the true observed responses.
- `y_lb::Union{Nothing, AbstractVector{<:Real}}`: Lower bounds of the output data. Defaults to `nothing`.
- `y_ub::Union{Nothing, AbstractVector{<:Real}}`: Upper bounds of the output data. Defaults to `nothing`.
"""
struct NormalizedData{
    XT<:AbstractMatrix{<:Real},
    YT<:AbstractMatrix{<:Real},
    YTO<:AbstractMatrix{<:Real},
    LB<:Union{Nothing, AbstractVector{<:Real}},
    UB<:Union{Nothing, AbstractVector{<:Real}},
} <: ExperimentData
    X::XT
    Y::YT
    Y_orig::YTO
    y_lb::LB
    y_ub::UB

    function NormalizedData(X::XT, Y::YT, Y_orig::YTO, y_lb::LB, y_ub::UB) where {XT, YT, YTO, LB, UB}
        @assert size(X, 2) == size(Y, 2) == size(Y_orig, 2)
        @assert size(Y_orig, 1) == size(Y, 1)
        @assert isnothing(y_lb) || length(y_lb) == size(Y, 1)
        @assert isnothing(y_ub) || length(y_ub) == size(Y, 1)
        @assert isnothing(y_lb) || isnothing(y_ub) || all(y_lb .< y_ub)
        return new{XT, YT, YTO, LB, UB}(X, Y, Y_orig, y_lb, y_ub)
    end
end

function NormalizedData(X::AbstractMatrix{<:Real}, Y_orig::AbstractMatrix{<:Real};
    y_lb = nothing,
    y_ub = nothing,
)
    Y = float.(Y_orig) # creates a deep copy
    scale_matrix!(Y, y_lb, y_ub)
    
    return NormalizedData(X, Y, Y_orig, y_lb, y_ub)
end

function NormalizedData(data::SimpleData; kwargs...)
    return NormalizedData(data.X, data.Y; kwargs...)
end

function scale_matrix!(M::AbstractMatrix{<:Real}, lb::Nothing, ub::Nothing)
    return M
end
function scale_matrix!(M::AbstractMatrix{<:Real}, lb::AbstractVector{<:Real}, ub::Nothing)
    (size(M, 2) == 0) && return M
    
    min_vals = minimum(M; dims=2)
    M .+= (lb .- min_vals)
    return M
end
function scale_matrix!(M::AbstractMatrix{<:Real}, lb::Nothing, ub::AbstractVector{<:Real})
    (size(M, 2) == 0) && return M
    
    max_vals = maximum(M; dims=2)
    M .+= (ub .- max_vals)
    return M
end
function scale_matrix!(M::AbstractMatrix{<:Real}, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real})
    (size(M, 2) == 0) && return M
    (size(M, 2) == 1) && return scale_matrix!(M, lb, nothing)
    
    min_vals = minimum(M; dims=2)
    max_vals = maximum(M; dims=2)
    scale = (ub .- lb) ./ (max_vals .- min_vals)
    M .*= scale
    M .+= lb .- min_vals .* scale
    return M
end

function augment_dataset(data::NormalizedData, X::AbstractArray{<:Real}, Y::AbstractArray{<:Real})
    X_ = hcat(data.X, X)
    Y_ = hcat(data.Y_orig, Y)

    return NormalizedData(X_, Y_;
        data.y_lb,
        data.y_ub,
    )
end

function update_dataset(data::NormalizedData, X::AbstractArray{<:Real}, Y::AbstractArray{<:Real})
    return NormalizedData(X, Y;
        data.y_lb,
        data.y_ub,
    )
end

function slice(data::NormalizedData, idx::Int)
    return NormalizedData(
        data.X,
        data.Y[idx:idx,:],
        data.Y_orig[idx:idx,:],
        isnothing(data.y_lb) ? nothing : data.y_lb[idx:idx],
        isnothing(data.y_ub) ? nothing : data.y_ub[idx:idx],
    )
end
