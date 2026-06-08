
# Plotting utilities for GP hyperparameter evolution.
# Requires: Plots, BOSS (for GaussianProcessParams, ParamsCallback, get_params)

"""
Extract a single `GaussianProcessParams` from a `FittedParams` entry.
For `MultiFittedParams` (e.g. BI samples), returns the element-wise mean.
"""
function _gp_params(fitted::FittedParams)
    p = get_params(fitted)
    if p isa GaussianProcessParams
        return p
    elseif p isa AbstractVector{<:GaussianProcessParams}
        n = length(p)
        λ = sum(x.λ for x in p) ./ n
        α = sum(x.α for x in p) ./ n
        σ = sum(x.σ for x in p) ./ n
        return GaussianProcessParams(λ, α, σ)
    else
        error("Expected GaussianProcessParams, got $(typeof(p))")
    end
end

"""
    plot_gp_hyperparameters(cb::ParamsCallback)

Plot the evolution of GP hyperparameters (λ, α, σ) recorded by a `ParamsCallback`.
Produces a three-panel figure with one series per output dimension.
Iteration 0 corresponds to the state before the BO loop starts.
"""
function plot_gp_hyperparameters(cb::ParamsCallback)
    gp_params = _gp_params.(cb.params_history)
    iters = 0:length(gp_params)-1
    y_dim = length(gp_params[1].α)
    labels = reshape(["y$i" for i in 1:y_dim], 1, :)

    p_λ = plot(iters,
        hcat([gp_params[t+1].λ[1, :] for t in iters]...)',
        xlabel = "iteration", ylabel = "λ (length scale)",
        title = "Length scales", label = labels,
    )
    p_α = plot(iters,
        hcat([gp_params[t+1].α for t in iters]...)',
        xlabel = "iteration", ylabel = "α (amplitude)",
        title = "Amplitudes", label = labels,
    )
    p_σ = plot(iters,
        hcat([gp_params[t+1].σ for t in iters]...)',
        xlabel = "iteration", ylabel = "σ (noise std)",
        title = "Noise std devs", label = labels,
    )

    return plot(p_λ, p_α, p_σ; layout = (3, 1), size = (800, 700))
end
