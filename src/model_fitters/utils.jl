
function _check_map_params(params::MAPParams; info::Bool)
    @assert !ismissing(params.loglike)
    
    if isinf(params.loglike)
        error("""Failed to find feasible model hyperparameters!
        This is a sign of numerical issues. Try to allow for higher data noise
        or reduce duplicities in the dataset to improve the regularity of the kernel matrix.""")
    else
        return params
    end
end
function _check_map_params(params::AbstractVector{<:MAPParams}; info::Bool)
    loglikes = getproperty.(params, Ref(:loglike))
    @assert !any(ismissing.(loglikes))

    if isinf(maximum(loglikes))
        error("""Failed to find feasible model hyperparameters!
        This is a sign of numerical issues. Try to allow for higher data noise
        or reduce duplicities in the dataset to improve the regularity of the kernel matrix.""")
    else
        return params
    end
end

function _check_bi_params(params::BIParams; info::Bool)  
    @assert !any(ismissing.(params.loglikes))
    infeasible = isinf.(params.loglikes)

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
            params.loglikes[feasible],
        )
    
    else
        return params
    end
end
