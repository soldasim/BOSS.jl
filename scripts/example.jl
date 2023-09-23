using BOSS
using Plots
using Distributions
using Random

using Turing
using OptimizationOptimJL

Random.seed!(555)

# We have an unknown noisy function 'blackbox(x)=y,z' and we want to maximize y s.t. z < 0 on domain x ∈ [0,20].

# The unknown blackbox function.
function blackbox(x; noise=0.1)
    y = exp(x[1]/10) * cos(2*x[1])
    z = (1/2)^6 * (x[1]^2 - (15.)^2)
    
    y += rand(Normal(0., noise))
    z += rand(Normal(0., noise))

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

    param_priors = fill(Normal(0., 1.), 3)

    BOSS.NonlinModel(; predict, param_priors)
end
# You can also try how the Parametric and Semiparametric models behave
# when we provide a completely wrong parametric model.
#
# Here we wrongly assume that the objective function will have a parabolic shape.
function bad_parametric_model()
    y(x, θ) = θ[1] * x[1]^2 + θ[2] * x[1] + θ[3]
    z(x, θ) = 0.
    predict(x, θ) = [y(x, θ), z(x, θ)]

    param_priors = fill(Normal(0., 1.), 3)

    BOSS.NonlinModel(predict, param_priors)
end

# Our prediction about the noise and GP length scales.
noise_var_priors() = fill(LogNormal(-2.3, 1.), 2)  # noise variance prior
# noise_var_priors() = fill(Dirac(0.1), 2)           # predefined noise variance value
length_scale_priors() = fill(MvLogNormal(0.1*ones(1), 1.0*ones(1)), 2)

# Generate some initial data.
function gen_data(count, bounds)
    X = reduce(hcat, [BOSS.random_start(bounds) for i in 1:count])
    Y = reduce(hcat, blackbox.(eachcol(X)))
    return X, Y
end

# The problem defined as `BOSS.OptimizationProblem`.
function opt_problem(init_data=4)
    domain = BOSS.Domain(;
        bounds = ([0.], [20.]),
    )
    data = gen_data(init_data, domain.bounds)

    # Try changing the parametric model here.
    model = BOSS.Semiparametric(
        good_parametric_model(),
        # bad_parametric_model(),
        BOSS.Nonparametric(; length_scale_priors=length_scale_priors())
    )

    BOSS.OptimizationProblem(;
        fitness = BOSS.LinFitness([1, 0]),
        f = blackbox,
        domain,
        y_max = [Inf, 0.],
        model,
        noise_var_priors = noise_var_priors(),
        data = BOSS.ExperimentDataPrior(data...),
    )
end

default_options = BOSS.BossOptions(;
    info=true,
    plot_options=BOSS.PlotOptions(Plots, f_true=x->blackbox(x; noise=0.)),
)

"""
An example usage of the BOSS algorithm with a MLE algorithm.
"""
function example_mle(problem=opt_problem(4), iters=3;
    parallel=true,
    options=default_options,
)
    # Algorithm selection and hyperparameters:
    model_fitter = BOSS.OptimizationMLE(;
        algorithm=NelderMead(),
        multistart=200,
        parallel,
    )

    acq_maximizer = BOSS.OptimizationAM(;
        algorithm=LBFGS(),
        multistart=200,
        parallel,
        x_tol=1e-2,
    )

    acquisition = BOSS.ExpectedImprovement()
    term_cond = BOSS.IterLimit(iters)

    # Run BOSS:
    boss!(problem; model_fitter, acq_maximizer, acquisition, term_cond, options)
end

"""
An example usage of the BOSS algorithm with a BI algorithm.
"""
function example_bi(problem=opt_problem(4), iters=3;
    parallel=true,
    options=default_options,
)
    # Algorithm selection and hyperparameters:
    model_fitter = BOSS.TuringBI(;
        sampler=PG(20),
        warmup=1000,
        samples_in_chain=10,
        chain_count=8,
        leap_size=5,
        parallel,
    )

    acq_maximizer = BOSS.OptimizationAM(;
        algorithm=LBFGS(),
        multistart=200,
        parallel,
        x_tol=1e-2,
    )

    acquisition = BOSS.ExpectedImprovement()
    term_cond = BOSS.IterLimit(iters)

    # Run BOSS:
    boss!(problem; model_fitter, acq_maximizer, acquisition, term_cond, options)
end
