using BOSS
using Plots
using Distributions
using Random

using OptimizationPRIMA
using Turing

Random.seed!(555)

# We have an unknown noisy function 'blackbox(x)=y,z' and we want to maximize y s.t. z < 0 on domain x ∈ [0,20].

# The unknown blackbox function.
function blackbox(x; noise_std=0.1)
    y = exp(x[1]/10) * cos(2*x[1])
    z = (1/2)^6 * (x[1]^2 - (15.)^2)
    
    y += rand(Normal(0., noise_std))
    z += rand(Normal(0., noise_std))

    return [y,z]
end

# Our parametric model represents our predictions/knowledge about the blackbox function.
# 
# Let's assume we know that x->y is a periodic function, so we use `cos` there.
# We don't know anything about x->z, so we put a constant 0 there.
# (That is equivalent to using a simple GP with zero-mean to model x->z.)
function good_parametric_model()
    y(x, θ) = θ[1] * x[1] * cos(θ[2] * x[1]) + θ[3]
    z(x, θ) = 0.
    predict(x, θ) = [y(x, θ), z(x, θ)]

    theta_priors = fill(Normal(0., 1.), 3)

    NonlinearModel(; predict, theta_priors)
end
# You can also try how the Parametric and Semiparametric models behave
# when we provide a completely wrong parametric model.
#
# Here we wrongly assume that the objective function will have a parabolic shape.
function bad_parametric_model()
    y(x, θ) = θ[1] * x[1]^2 + θ[2] * x[1] + θ[3]
    z(x, θ) = 0.
    predict(x, θ) = [y(x, θ), z(x, θ)]

    theta_priors = fill(Normal(0., 1.), 3)

    NonlinearModel(; predict, theta_priors)
end

# Our prediction about the noise and GP hyperparameters.
# - - A) Predefined values - - - -
# lengthscale_priors() = fill(Product(fill(Dirac(1.), 1)), 2)
# amplitude_priors() = fill(Dirac(1.), 2)
# noise_std_priors() = fill(Dirac(0.1), 2)
# - - B) Priors - - - -
lengthscale_priors() = fill(Product([truncated(Normal(0., 20/3); lower=0.)]), 2)
amplitude_priors() = fill(truncated(Normal(0., 10.); lower=0.), 2)
noise_std_priors() = fill(truncated(Normal(0., 1.); lower=0.), 2)

# Generate some initial data.
function gen_data(count, bounds)
    X = reduce(hcat, [BOSS.random_point(bounds) for i in 1:count])
    Y = reduce(hcat, blackbox.(eachcol(X)))
    return X, Y
end

# The problem defined as `BossProblem`.
function opt_problem(init_data)
    domain = Domain(;
        bounds = ([0.], [20.]),
    )
    data = gen_data(init_data, domain.bounds)

    # Try using the Semiparametric model and changing the parametric mean.
    # model = Semiparametric(
    #     good_parametric_model(),
    #     # bad_parametric_model(),
    #     Nonparametric(;
    #         kernel = BOSS.Matern32Kernel(),
    #         amplitude_priors = amplitude_priors(),
    #         lengthscale_priors = lengthscale_priors(),
    #         noise_std_priors = noise_std_priors(),
    #     ),
    # )
    model = Nonparametric(;
        kernel = BOSS.Matern32Kernel(),
        amplitude_priors = amplitude_priors(),
        lengthscale_priors = lengthscale_priors(),
        noise_std_priors = noise_std_priors(),
    )

    BossProblem(;
        fitness = LinFitness([1, 0]),
        f = blackbox,
        domain,
        y_max = [Inf, 0.],
        model,
        data = ExperimentData(data...),
    )
end

boss_options() = BossOptions(;
    info = true,
    debug = false,
    callback = PlotCallback(Plots;
        f_true = x->blackbox(x; noise_std=0.),
    ),
)

"""
An example usage of the BOSS algorithm with a MAP algorithm.
"""
function main(problem=opt_problem(3), iters=20;
    parallel = true,
)
    ### Model Fitter:
    # Maximum likelihood estimation
    model_fitter = OptimizationMAP(;
        algorithm = NEWUOA(),
        multistart = 20,
        parallel,
        rhoend = 1e-4,
    )
    # # Bayesian Inference (sampling)
    # model_fitter = TuringBI(;
    #     sampler = NUTS(20, 0.65),
    #     warmup = 100,
    #     samples_in_chain = 25,
    #     chain_count = 4,
    #     leap_size = 5,
    #     parallel,
    # )

    ### Acquisition Maximizer:
    acq_maximizer = OptimizationAM(;
        algorithm = BOBYQA(),
        multistart = 20,
        parallel,
        rhoend = 1e-4,
    )

    acquisition = ExpectedImprovement()
    term_cond = IterLimit(iters)
    options = boss_options()

    # Run BOSS:
    bo!(problem; model_fitter, acq_maximizer, acquisition, term_cond, options)
end
