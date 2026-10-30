using BOSS
using Distributions
using Plots
using Random
using OptimizationPRIMA
using Turing

Random.seed!(555)

# Maximize y s.t. z < 0 for the unknown noisy function `blackbox(x) = [y, z]` on x ∈ [0, 20].
function blackbox(x; noise_std=0.0)
    y = exp(x[1]/10) * cos(2*x[1]) + rand(Normal(0., noise_std))
    z = (1/2)^6 * (x[1]^2 - 15^2) + rand(Normal(0., noise_std))
    return [y, z]
end

# Prior knowledge: x->y is periodic, nothing is known about x->z.
parametric_model() = NonlinearModel(;
    predict = (x, θ) -> [θ[1] * x[1] * cos(θ[2] * x[1]) + θ[3], 0.],
    theta_priors = fill(Normal(0., 1.), 3),
)

# Gaussian process model
gp_model() = GaussianProcess(;
    kernel = BOSS.Matern32Kernel(),
    amplitude_priors = fill(truncated(Normal(0., 10.); lower=0., upper=100.), 2),
    lengthscale_priors = fill(Product([truncated(Normal(0., 20/3); lower=0., upper=20.)]), 2),
    noise_std_priors = fill(Dirac(0.), 2),
)

# Define the optimization problem with optional prior knowledge.
function opt_problem(init_data_count; use_prior_knowledge=true)
    domain = Domain(; bounds = ([0.], [20.]))

    X = reduce(hcat, [BOSS.random_point(domain.bounds) for _ in 1:init_data_count])
    Y = reduce(hcat, blackbox.(eachcol(X)))

    if use_prior_knowledge
        model = Semiparametric(;
            parametric = parametric_model(),
            nonparametric = gp_model(),
        )
    else
        model = gp_model()
    end

    BossProblem(;
        f = blackbox,
        domain,
        y_max = [Inf, 0.],
        acquisition = ExpectedImprovement(; fitness = LinFitness([1, 0])),
        model,
        data = ExperimentData(X, Y),
    )
end

"""
Run BOSS on the example problem and return the solved `problem`.

Use `result(problem)` to get the best solution `(x, y)`,
and `continue!(problem)` to run more iterations.

## Keywords
- `init_data_count`: Number of initial random points to sample (default: 3)
- `iters`: Number of BOSS iterations to run (default: 20)
- `use_prior_knowledge`: Whether to use prior knowledge in the model (default: true)
- `sample_hyperparams`: Whether to sample hyperparameters using MCMC (default: false)
- `parallel`: Whether to run model fitting and acquisition maximization in parallel (default: false)

Note that there are issues with parallelization with PRIMA algorithms on Linux: https://github.com/libprima/PRIMA.jl/issues/25
"""
function main(; init_data_count=3, iters=20, use_prior_knowledge=true, sample_hyperparams=false, parallel=false)
    problem = opt_problem(init_data_count; use_prior_knowledge)
    continue!(problem; iters, sample_hyperparams, parallel)
end

make_model_fitter(::Val{false}; parallel) = OptimizationMAP(;
    algorithm = NEWUOA(),
    multistart = 20,
    parallel,
    rhoend = 1e-4,
)
make_model_fitter(::Val{true}; parallel) = TuringBI(;
    sampler = NUTS(20, 0.65),
    warmup = 200,
    samples_in_chain = 10,
    chain_count = 12,
    leap_size = 5,
    parallel,
)

"""
Run more BOSS iterations on an existing `problem` (it keeps all data collected so far).
"""
function continue!(problem; iters=10, sample_hyperparams=false, parallel=false)
    if sample_hyperparams
        model_fitter = TuringBI(;
            sampler = NUTS(20, 0.65),
            warmup = 200,
            samples_in_chain = 10,
            chain_count = 12,
            leap_size = 5,
            parallel,
        )
    else
        model_fitter = OptimizationMAP(;
            algorithm = NEWUOA(),
            multistart = 20,
            parallel,
            rhoend = 1e-4,
        )
    end

    acq_maximizer = OptimizationAM(;
        algorithm = BOBYQA(),
        multistart = 20,
        parallel,
        rhoend = 1e-4,
    )

    # Add a callback for plotting the iterations
    options = BossOptions(;
        callback = PlotCallback(Plots; f_true = x -> blackbox(x; noise_std=0.)),
    )

    bo!(problem; model_fitter, acq_maximizer, term_cond = IterLimit(iters), options)

    x, y = result(problem)
    @info "Best solution found: x = $x, y = $y"
    return problem
end


# Run the example problem
problem = main()
