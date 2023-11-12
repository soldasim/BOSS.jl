
"""
Return an averaged posterior predictive distribution of the given posteriors.

The posterior is a function `mean, var = predict(x)`
which gives the mean and variance of the predictive distribution as a function of `x`.
"""
average_posteriors(posteriors::AbstractVector{<:Function}) =
    x -> mapreduce(p -> p(x), .+, posteriors) ./ length(posteriors)

discrete_round(::Nothing, x::AbstractVector{<:Real}) = x
discrete_round(::Missing, x::AbstractVector{<:Real}) = round.(x)
discrete_round(dims::AbstractVector{<:Bool}, x::AbstractVector{<:Real}) = cond_func(round).(dims, x)

noise_loglike(noise_var_priors, noise_vars) = mapreduce(p -> logpdf(p...), +, zip(noise_var_priors, noise_vars))
