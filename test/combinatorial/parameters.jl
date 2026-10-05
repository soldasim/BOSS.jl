## The options for each parameter used in the combinatorial tests are defined here.

function OBJECTIVE(x; noise_std=0.05)
    y = exp(x[1] / 10) * cos(2 * x[1]) + rand(Normal(0., noise_std))
    z = (1 / 2)^6 * (x[1]^2 - 15.0^2) + rand(Normal(0., noise_std))
    return [y, z]
end

const BOUNDS = ([0., 0.], [20., 20.])

xy_data(X) = (X, reduce(hcat, OBJECTIVE.(eachcol(X))))
const XY_OPTIONS = [
    xy_data([5. 10. 10.01; 15. 2. 2.]),  # near-duplicate x-values (offset to avoid a ForwardDiff NaN at the kernel's zero-distance singularity; models needing higher-order kernel derivatives, e.g. GradientGP, need a larger margin than 1e-4)
    xy_data([5. 10. 15.; 15. 2. 8.]),  # no duplicates
]

# `OBJECTIVE`'s analytic Jacobian, for `GradientGP` (ignores x[2], which
# `OBJECTIVE` doesn't depend on).
function OBJECTIVE_GRAD(x; noise_std=0.05)
    y = exp(x[1] / 10) * cos(2 * x[1])
    z = (1 / 2)^6 * (x[1]^2 - 15.0^2)
    dy_dx1 = exp(x[1] / 10) * (cos(2 * x[1]) / 10 - 2 * sin(2 * x[1]))
    dz_dx1 = x[1] / 32
    y_noisy = y + rand(Normal(0., noise_std))
    z_noisy = z + rand(Normal(0., noise_std))
    J = [dy_dx1 0.; dz_dx1 0.]  # y_dim × x_dim
    return [y_noisy, z_noisy], J
end

function gradient_data(X)
    results = OBJECTIVE_GRAD.(eachcol(X))
    Y = reduce(hcat, first.(results))
    dY = cat(last.(results)...; dims=3)
    return GradientData(X, Y, dY)
end

const F_OPTIONS = [OBJECTIVE, missing]
const DISCRETE_OPTIONS = [[false, false], [true, false]]
const CONS_OPTIONS = [(x) -> [x[1] - 5.], nothing]
const Y_MAX_OPTIONS = [[Inf, 0.], [Inf, Inf]]
const ITER_MAX_OPTIONS = [1, 2]

const FITNESS_OPTIONS = [
    LinFitness([1., 0.]),
    NonlinFitness(y -> y[1]),
]

const PARAMETRIC_PREDICT = (x, θ) -> [θ[1] * x[1] * cos(θ[2] * x[1]) + θ[3], 0.]
const NONPARAMETRIC_MEAN = (x) -> [cos(2 * x[1]), 0.]
const NONPARAMETRIC_KERNEL = Matern32Kernel()

const THETA_PRIORS_WITH_DIRAC = [Normal(0., 1.), Normal(0., 1.), Dirac(0.)]
const THETA_PRIORS_PLAIN = fill(Normal(0., 1.), 3)

const AMPLITUDE_PRIORS_WITH_DIRAC = fill(Dirac(1.), 2)
const AMPLITUDE_PRIORS_PLAIN = scaled_half_t_prior(2., 2)  # rough a priori amplitude guess, not fitted to OBJECTIVE

const LENGTHSCALE_PRIORS_WITH_DIRAC = fill(product_distribution(fill(Dirac(1.), 2)), 2)
const LENGTHSCALE_PRIORS_PLAIN = lengthscale_prior(BOUNDS, 2)

const NOISE_STD_PRIORS_WITH_DIRAC = fill(Dirac(0.1), 2)
const NOISE_STD_PRIORS_PLAIN = scaled_half_t_prior(0.05, 2)

const GRAD_NOISE_STD_PRIORS_PLAIN = scaled_half_t_prior(0.05, 2)  # matches OBJECTIVE_GRAD's gradient noise_std

const MODEL_CONFIGS = [
    ModelConfig(:parametric_dirac, parametric_model(THETA_PRIORS_WITH_DIRAC, NOISE_STD_PRIORS_WITH_DIRAC)),
    ModelConfig(:parametric_plain, parametric_model(THETA_PRIORS_PLAIN, NOISE_STD_PRIORS_PLAIN)),
    ModelConfig(:nonparametric_dirac, nonparametric_model(NONPARAMETRIC_MEAN, AMPLITUDE_PRIORS_WITH_DIRAC, LENGTHSCALE_PRIORS_WITH_DIRAC, NOISE_STD_PRIORS_WITH_DIRAC)),
    ModelConfig(:nonparametric_plain, nonparametric_model(nothing, AMPLITUDE_PRIORS_PLAIN, LENGTHSCALE_PRIORS_PLAIN, NOISE_STD_PRIORS_PLAIN)),
    ModelConfig(:semiparametric, semiparametric_model(THETA_PRIORS_PLAIN, NONPARAMETRIC_MEAN, AMPLITUDE_PRIORS_WITH_DIRAC, LENGTHSCALE_PRIORS_PLAIN, NOISE_STD_PRIORS_PLAIN)),
    ModelConfig(:warped, warped_model(LENGTHSCALE_PRIORS_PLAIN, NOISE_STD_PRIORS_PLAIN)),
    ModelConfig(:nonstationary, nonstationary_model(LENGTHSCALE_PRIORS_PLAIN, AMPLITUDE_PRIORS_PLAIN, NOISE_STD_PRIORS_PLAIN)),
    ModelConfig(:gradient,
        gradient_model(LENGTHSCALE_PRIORS_PLAIN, AMPLITUDE_PRIORS_PLAIN, NOISE_STD_PRIORS_PLAIN, GRAD_NOISE_STD_PRIORS_PLAIN),
        (X, Y) -> gradient_data(X),
        f -> ismissing(f) ? missing : OBJECTIVE_GRAD,
        _ -> [false, false],  # GradientGP has no `make_discrete` method
    ),
]

const MODEL_FITTERS = [
    () -> SamplingMAP(; samples=10, parallel=PARALLEL_TESTS),               # low sample count to improve test runtime
    () -> OptimizationMAP(; algorithm=NEWUOA(), multistart=2, rhoend=1e-2, parallel=PARALLEL_TESTS),
    () -> TuringBI(; sampler=NUTS(20, 0.65), warmup=10, samples_in_chain=2, chain_count=2, leap_size=3, parallel=PARALLEL_TESTS),
    () -> RandomFitter(),
]

const ACQ_MAXIMIZERS = [
    (cons, problem) -> SamplingAM(; x_prior=product_distribution(Uniform.(BOUNDS...)), samples=10, parallel=PARALLEL_TESTS),
    (cons, problem) -> OptimizationAM(; algorithm=isnothing(cons) ? BOBYQA() : COBYLA(), multistart=2, rhoend=1e-2, parallel=PARALLEL_TESTS),
    (cons, problem) -> GridAM(; problem, steps=[1.], parallel=PARALLEL_TESTS),
    (cons, problem) -> RandomAM(),
]

"""
Generate the pairwise-covering set of test cases, each a `(case, closure)`
pair ready to be run and checked.
"""
function generate_test_cases()
    combinations = all_pairs(
        XY_OPTIONS, F_OPTIONS, DISCRETE_OPTIONS, CONS_OPTIONS, Y_MAX_OPTIONS,
        FITNESS_OPTIONS, MODEL_CONFIGS, MODEL_FITTERS, ACQ_MAXIMIZERS, ITER_MAX_OPTIONS,
    )
    return [(case, build_case(case...)) for case in combinations]
end
