
# Used for deprecating keywords.
struct NoVal end


# ExperimentDataPrior

export ExperimentDataPrior

function ExperimentDataPrior(args...; kwargs...)
    Base.depwarn("`ExperimentDataPrior` is deprecated. Use `ExperimentData` instead.", :ExperimentDataPrior; force=true)
    return ExperimentData(args...; kwargs...)
end


# LinModel, NonlinModel

export LinModel, NonlinModel

function LinModel(args...; kwargs...)
    Base.depwarn("`LinModel` is deprecated. Use `LinearModel` instead.", :LinModel; force=true)
    return LinearModel(args...; kwargs...)
end
function NonlinModel(args...; kwargs...)
    Base.depwarn("`NonlinModel` is deprecated. Use `NonlinearModel` instead.", :NonlinModel; force=true)
    return NonlinearModel(args...; kwargs...)
end


# GaussianProcess -> length_scale_priors, amp_priors

function GaussianProcess(;
    mean = nothing,
    kernel = Matern32Kernel(),
    length_scale_priors = NoVal, lengthscale_priors = NoVal,
    amp_priors = NoVal, amplitude_priors = NoVal,
    noise_std_priors,
)
    if length_scale_priors !== NoVal
        Base.depwarn("`length_scale_priors` is deprecated. Use `lengthscale_priors` instead.", :length_scale_priors; force=true)
        lengthscale_priors = length_scale_priors
    end
    if lengthscale_priors === NoVal
        throw(UndefKeywordError(:lengthscale_priors))
    end

    if amp_priors !== NoVal
        Base.depwarn("`amp_priors` is deprecated. Use `amplitude_priors` instead.", :amp_priors; force=true)
        amplitude_priors = amp_priors
    end
    if amplitude_priors === NoVal
        throw(UndefKeywordError(:amplitude_priors))
    end

    return GaussianProcess(
        mean,
        kernel,
        lengthscale_priors,
        amplitude_priors,
        noise_std_priors,
    )
end


# RandomMAP

export RandomMAP

function RandomMAP(args...; kwargs...)
    Base.depwarn("`RandomMAP` is deprecated. Use `RandomFitter` instead.", :RandomMAP; force=true)
    return RandomFitter(args...; kwargs...)
end
