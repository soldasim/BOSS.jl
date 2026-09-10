
function _check_map_params(params::MAPParams; info::Bool)
    @assert !ismissing(params.logpost)
    
    if isinf(params.logpost)
        error("""Failed to find feasible model hyperparameters!
        This is a sign of numerical issues. Try to allow for higher data noise
        or reduce duplicities in the dataset to improve the regularity of the kernel matrix.""")
    else
        return params
    end
end
function _check_map_params(params::AbstractVector{<:MAPParams}; info::Bool)
    logposts = getproperty.(params, Ref(:logpost))
    @assert !any(ismissing.(logposts))

    if isinf(maximum(logposts))
        error("""Failed to find feasible model hyperparameters!
        This is a sign of numerical issues. Try to allow for higher data noise
        or reduce duplicities in the dataset to improve the regularity of the kernel matrix.""")
    else
        return params
    end
end

function _check_bi_params(params::BIParams; info::Bool)  
    @assert !any(ismissing.(params.logposts))
    infeasible = isinf.(params.logposts)

    if all(infeasible)
        error("""Failed to sample any feasible model hyperparameters!
        This is a sign of numerical issues. Try to allow for higher data noise
        or reduce duplicities in the dataset to improve the regularity of the kernel matrix.
        Alternatively, there might be numerical issues in the sampler.""")
    
    elseif any(infeasible)
        count = sum(infeasible)
        total = length(infeasible)
        info && @warn "Dropping $count/$total hyperparameter samples due to numerical issues!"
        feasible = .! infeasible
        return BIParams(
            params.samples[feasible],
            params.logposts[feasible],
        )
    
    else
        return params
    end
end
