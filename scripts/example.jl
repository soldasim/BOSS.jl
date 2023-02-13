using BOSS
using Distributions
using Optim
using Turing
using Plots
using Random

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
function parametric_model()
    y(x, θ) = θ[1] * x[1] * cos(θ[2] * x[1]) + θ[3]
    z(x, θ) = 0.
    predict(x, θ) = [y(x, θ), z(x, θ)]

    θ_priors = [Normal(0., 1.), Normal(0., 1.), Normal(0., 10.)]

    BOSS.NonlinModel(predict, θ_priors)
end

# Our prediction about the noise and GP length scales.
noise_var_priors() = fill(LogNormal(-2.3, 1.), 2)
length_scale_priors() = fill(MvLogNormal(0.1*ones(1), 1.0*ones(1)), 2)

# Initial data.
function gen_data(count, bounds)
    X = reduce(hcat, [BOSS.random_start(bounds) for i in 1:count])
    Y = reduce(hcat, blackbox.(eachcol(X)))
    return X, Y
end

# The problem defined as `BOSS.OptimizationProblem`.
function opt_problem(init_data=4)
    bounds = ([0.], [20.])
    data = gen_data(init_data, bounds)

    # You can try out the parametric, nonparametric (GP) and semiparametric (param. + GP) models:
    model = BOSS.Semiparametric(parametric_model(), BOSS.Nonparametric(; length_scale_priors=length_scale_priors()))
    # model = BOSS.Nonparametric(; length_scale_priors=length_scale_priors())
    # model = parametric_model()

    BOSS.OptimizationProblem(;
        fitness = BOSS.LinFitness([1, 0]),
        f = blackbox,
        cons = [Inf, 0.],
        domain = BOSS.OptimDomain(bounds),
        discrete = [false],
        model,
        noise_var_priors = noise_var_priors(),
        data = BOSS.ExperimentDataPrior(data...),
    )
end

function example(problem=opt_problem(), iters=3)
    # Algorithm selection and hyperparameters:
    # You can try changing some. :)

    model_fitter = BOSS.OptimMLE(;
        # algorithm=Fminbox(LBFGS()),
        algorithm=NelderMead(),
        options=Optim.Options(; outer_x_tol=0.01),
        multistart=200,
        parallel=true,
    )
    # model_fitter = BOSS.TuringBI(;
    #     # sampler=NUTS(1000, 0.65),
    #     sampler=Turing.PG(20),
    #     warmup=400,
    #     samples_in_chain=10,
    #     chain_count=8,
    #     leap_size=5,
    #     parallel=true,
    # )

    acq_maximizer = BOSS.OptimMaximizer(;
        algorithm=Fminbox(LBFGS()),
        options=Optim.Options(; outer_x_tol=0.01),
        multistart=200,
        parallel=true,
    )
    
    term_cond = BOSS.IterLimit(iters)
    options = BOSS.BossOptions(;
        info=true,
        plot_options=BOSS.PlotOptions(Plots, f_true=x->blackbox(x; noise=0.)),
    )

    # Run BOSS:
    boss!(problem; model_fitter, acq_maximizer, term_cond, options)
end
