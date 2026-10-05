"""
Constructor-like helper functions used to build the combinatorial test
parameters in `parameters.jl`.
"""

"""
A domain-scale-aware lengthscale prior, ported from `bosip_benchmarks`'
`param_priors.jl`: a per-dimension `LogNormal` whose 2-std range spans
`[domain_size/20, domain_size]`, truncated above by the domain size so GP
hyperparameter search can't wander into a lengthscale so large (relative to
the domain) that the kernel matrix becomes near-singular.
"""
function lengthscale_prior(bounds, y_dim)
    lb, ub = bounds
    d = ub .- lb
    min_λ = d ./ 20
    max_λ = d
    μ = (log.(min_λ) .+ log.(max_λ)) ./ 2
    σ = (log.(max_λ) .- log.(min_λ)) ./ 2
    dists = map((d, m) -> truncated(d; upper=m), LogNormal.(μ, σ), max_λ)
    dist = product_distribution(dists)
    return fill(dist, y_dim)
end

"""
A half-t-distributed prior scaled to a rough per-output magnitude estimate,
ported from `bosip_benchmarks`' `param_priors.jl`. Used for both
`amplitude_priors` and `noise_std_priors`. `est_scale` is either one shared
scale (a `Real`) or one estimate per output dimension (an `AbstractVector`).
"""
scaled_half_t_prior(est_scale::Real, y_dim) = fill(scaled_half_t_prior(est_scale), y_dim)
scaled_half_t_prior(est_scale::AbstractVector{<:Real}) = scaled_half_t_prior.(est_scale)
scaled_half_t_prior(est_scale::Real) = transformed(truncated(TDist(2); lower=0.), Bijectors.Scale(est_scale))

"""
One `ModelConfig` fully specifies a surrogate model, bundling it with
everything that only makes sense in combination with it: its mean, kernel,
and hyperparameter priors (including `noise_std_priors`).

`build_data`/`build_f`/`build_discrete` let a model override how the
problem's `data`/`f`/`discrete` are built from the shared
`XY_OPTIONS`/`F_OPTIONS`/`DISCRETE_OPTIONS` draws, for models that need a
different contract (e.g. `GradientGP` needs `GradientData`, a
gradient-returning objective, and doesn't support discrete dims at all — it
has no `make_discrete` method). They default to a plain pass-through for
every other model.
"""
struct ModelConfig
    name::Symbol
    model::SurrogateModel
    build_data::Function      # (X, Y) -> ExperimentData
    build_f::Function         # f -> f (or an objective matching the model's data contract)
    build_discrete::Function  # discrete -> discrete
end
ModelConfig(name, model) = ModelConfig(name, model, (X, Y) -> ExperimentData(X, Y), identity, identity)

parametric_model(theta_priors, noise_std_priors; no_noise=false) =
    NonlinearModel(;
        predict = PARAMETRIC_PREDICT,
        theta_priors,
        noise_std_priors = no_noise ? nothing : noise_std_priors,
    )

nonparametric_model(mean, amplitude_priors, lengthscale_priors, noise_std_priors) =
    Nonparametric(;
        mean,
        kernel = NONPARAMETRIC_KERNEL,
        amplitude_priors,
        lengthscale_priors,
        noise_std_priors,
    )

semiparametric_model(parametric_theta_priors, mean, amplitude_priors, lengthscale_priors, noise_std_priors) =
    Semiparametric(;
        parametric = parametric_model(parametric_theta_priors, noise_std_priors; no_noise=true),
        nonparametric = nonparametric_model(mean, amplitude_priors, lengthscale_priors, noise_std_priors),
    )

warped_model(lengthscale_priors, noise_std_priors) =
    WarpedGP(; lengthscale_priors, noise_std_priors)
    # default amplitude fixed to 1: redundant with the trailing AffineWarping's scale.

nonstationary_model(mean, lengthscale_model, amplitude_model, noise_std_model) =
    NonstationaryGP(; mean, lengthscale_model, amplitude_model, noise_std_model)
    # Uses the plain priors-vector form (like `GaussianProcess`), not the
    # `ParametrizedGP`-matrix nonstationary-lengthscale form: that form only
    # supports fully-fixed (`Dirac`) lengthscale priors for now (its non-Dirac
    # `params_logprior` branch is an unimplemented stub).

gradient_model(mean, lengthscale_priors, amplitude_priors, noise_std_priors, grad_noise_std_priors) =
    GradientGP(; mean, lengthscale_priors, amplitude_priors, noise_std_priors, grad_noise_std_priors)

"""
Build a runnable test case (a closure calling `bo!`) from one all-pairs
combination of the parameters in `parameters.jl`.
"""
function build_case(XY, f, discrete, cons, y_max, fitness, model_config, make_fitter, make_maximizer, iter_max)
    X, Y = XY
    f = model_config.build_f(f)
    discrete = model_config.build_discrete(discrete)

    problem = BossProblem(;
        f,
        domain = Domain(; bounds=BOUNDS, discrete, cons),
        acquisition = ExpectedImprovement(; fitness),
        model = model_config.model,
        y_max,
        data = model_config.build_data(X, Y),
    )
    model_fitter = make_fitter()
    acq_maximizer = make_maximizer(cons, problem)
    options = BossOptions(; info=true, debug=false)

    if ismissing(f)
        return () -> bo!(problem; model_fitter, acq_maximizer, options)
    else
        term_cond = IterLimit(iter_max)
        return () -> bo!(problem; model_fitter, acq_maximizer, term_cond, options)
    end
end
