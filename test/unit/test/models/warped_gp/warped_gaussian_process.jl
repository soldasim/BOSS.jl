
# An identity warping (per output dimension), so the warped GP reduces to a plain GP.
# Parameters are fixed via `Dirac` priors so they drop out of the free parameter vector.
_identity_warps(y_dim) =
    [AffineWarping(; shift_prior=Dirac(0.), scale_prior=Dirac(1.)) for _ in 1:y_dim]

# A warped GP and a plain GP sharing the same GP hyperprior configuration.
function _twin_models(y_dim; mean=nothing)
    kwargs = (;
        mean,
        amplitude_priors = fill(LogNormal(), y_dim),
        lengthscale_priors = fill(BOSS.mvlognormal(fill(1., 2), fill(1., 2)), y_dim),
        noise_std_priors = fill(Dirac(1e-4), y_dim),
    )
    warped = WarpedGaussianProcess(; kwargs..., output_warpings=_identity_warps(y_dim))
    plain = GaussianProcess(; kwargs...)
    return warped, plain
end

@testset "make_discrete(model, discrete)" begin
    @param_test BOSS.make_discrete begin
        @params (
            WarpedGaussianProcess(;
                kernel = Matern32Kernel(),
                amplitude_priors = fill(LogNormal(), 2),
                lengthscale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
                noise_std_priors = fill(Dirac(0.1), 2),
                output_warpings = _identity_warps(2),
            ),
            [false, true],
        )
        @success (
            out isa WarpedGaussianProcess,
            out.kernel isa BOSS.DiscreteKernel,
            out.kernel.kernel == in[1].kernel,
            out.output_warpings == in[1].output_warpings,
            out.quad_nodes == in[1].quad_nodes,
            out.kernel([1.2, 1.2], [3.8, 3.8]) == out.kernel([1.2, 1.], [3.8, 4.]),
        )
    end
end

@testset "sliceable / slice / join_slices" begin
    model = WarpedGaussianProcess(;
        amplitude_priors = fill(LogNormal(), 2),
        lengthscale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
        noise_std_priors = fill(Dirac(0.1), 2),
        output_warpings = [
            YeoJohnsonWarping(; λ_prior=truncated(Normal(1, 0.5); lower=0, upper=2)),
            ComposedWarping(
                YeoJohnsonWarping(; λ_prior=truncated(Normal(1, 0.5); lower=0, upper=2)),
                AffineWarping(; shift_prior=Normal(), scale_prior=LogNormal()),
            ),
        ],
    )
    params = WarpedGaussianProcessParams(
        [1.;1.;; 2.;2.;;], [1., 2.], [0.1, 0.2], [[0.5], [0.4, 0.1, 1.3]],
    )

    @test BOSS.sliceable(model)

    s1 = BOSS.slice(model, 1)
    s2 = BOSS.slice(model, 2)
    @test s1 isa WarpedGaussianProcess
    @test length(s1.output_warpings) == 1
    @test s1.output_warpings[1] === model.output_warpings[1]
    @test s2.output_warpings[1] === model.output_warpings[2]

    p1 = BOSS.slice(params, 1)
    p2 = BOSS.slice(params, 2)
    @test p1.warp == [[0.5]]
    @test p2.warp == [[0.4, 0.1, 1.3]]
    @test size(p1.λ) == (2, 1)

    # join_slices is the inverse of slicing the parameters.
    joined = BOSS.join_slices([p1, p2])
    @test joined.λ == params.λ
    @test joined.α == params.α
    @test joined.σ == params.σ
    @test joined.warp == params.warp
end

@testset "vectorizer round-trip and bijector" begin
    model = WarpedGaussianProcess(;
        amplitude_priors = fill(LogNormal(), 2),
        lengthscale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
        noise_std_priors = fill(Dirac(0.1), 2),  # fixed -> excluded from free params
        output_warpings = [
            YeoJohnsonWarping(; λ_prior=truncated(Normal(1, 0.5); lower=0, upper=2)),
            ComposedWarping(
                YeoJohnsonWarping(; λ_prior=truncated(Normal(1, 0.5); lower=0, upper=2)),
                AffineWarping(; shift_prior=Normal(), scale_prior=LogNormal()),
            ),
        ],
    )
    params = WarpedGaussianProcessParams(
        [1.;1.;; 2.;2.;;], [1., 2.], [0.1, 0.1], [[0.5], [0.4, 0.1, 1.3]],
    )
    vectorize, devectorize = BOSS.vectorizer(model)
    v = vectorize(params)

    @test v isa AbstractVector{<:Real}
    # free params: 4 lengthscales + 2 amplitudes + 0 noise (Dirac) + 4 warp params = 10
    @test length(v) == 10

    rt = devectorize(params, v)
    @test rt.λ == params.λ
    @test rt.α == params.α
    @test rt.σ == params.σ          # fixed noise restored
    @test rt.warp == params.warp
    @test vectorize(rt) == v

    b = BOSS.bijector(model)
    @test length(b(v)) == length(v)
    @test isapprox(BOSS.Bijectors.inverse(b)(b(v)), v; atol=1e-8)
end

@testset "data_loglike: identity warp matches plain GaussianProcess" begin
    warped, plain = _twin_models(1)
    X = [1.;; 2.;; 3.;;]
    Y = [0.5;; -1.2;; 2.1;;]
    data = ExperimentData(X, Y)

    wll = BOSS.data_loglike(warped, data)
    pll = BOSS.data_loglike(plain, data)

    gp_params = GaussianProcessParams([1.;;], [1.], [0.5])
    w_params = WarpedGaussianProcessParams([1.;;], [1.], [0.5], [[0.0, 1.0]])

    @test wll(w_params) isa Real
    @test isapprox(wll(w_params), pll(gp_params); atol=1e-8)
end

@testset "data_loglike: Jacobian rewards the correct warp on log-scale data" begin
    # Strictly-positive, log-scale data: a log-like warp (Yeo-Johnson λ→0) should achieve
    # higher marginal likelihood than the identity (λ=1). This only holds because the
    # Jacobian correction `Σ log|φ'(y)|` is included.
    model = WarpedGaussianProcess(;
        mean = x -> [0.],
        amplitude_priors = fill(LogNormal(), 1),
        lengthscale_priors = fill(product_distribution(fill(Dirac(1.), 1)), 1),
        noise_std_priors = fill(Dirac(1e-2), 1),
        output_warpings = [YeoJohnsonWarping(; λ_prior=truncated(Normal(1, 1); lower=0, upper=2))],
    )
    X = [1.;; 2.;; 3.;; 4.;;]
    Y = [exp(2.0);; exp(4.0);; exp(6.0);; exp(8.0);;]  # spans orders of magnitude
    ll = BOSS.data_loglike(model, ExperimentData(X, Y))

    ll_log = ll(WarpedGaussianProcessParams([1.;;], [1.], [1e-2], [[0.0]]))   # ~log warp
    ll_id  = ll(WarpedGaussianProcessParams([1.;;], [1.], [1e-2], [[1.0]]))   # identity
    @test ll_log > ll_id
end

@testset "params_logprior(model, params)" begin
    @param_test BOSS.params_logprior begin
        @params WarpedGaussianProcess(;
            lengthscale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
            amplitude_priors = fill(LogNormal(), 2),
            noise_std_priors = fill(Dirac(0.1), 2),
            output_warpings = [
                YeoJohnsonWarping(; λ_prior=truncated(Normal(1, 0.5); lower=0, upper=2)),
                YeoJohnsonWarping(; λ_prior=truncated(Normal(1, 0.5); lower=0, upper=2)),
            ],
        )
        @success (
            out(WarpedGaussianProcessParams([1.;1.;; 1.;1.;;], [1., 2.], [0.1, 0.1], [[1.0], [1.0]])) isa Real,
            # warp prior contributes: λ near the prior mean beats λ far from it
            out(WarpedGaussianProcessParams([1.;1.;; 1.;1.;;], [1., 1.], [0.1, 0.1], [[1.0], [1.0]])) >
                out(WarpedGaussianProcessParams([1.;1.;; 1.;1.;;], [1., 1.], [0.1, 0.1], [[1.0], [5.0]])),
        )

        # Fixed (Dirac) warp param outside its support => -Inf
        @params WarpedGaussianProcess(;
            lengthscale_priors = fill(product_distribution(fill(Dirac(1.), 2)), 2),
            amplitude_priors = fill(Dirac(1.), 2),
            noise_std_priors = fill(Dirac(0.1), 2),
            output_warpings = [
                AffineWarping(; shift_prior=Dirac(0.), scale_prior=Dirac(1.)),
                AffineWarping(; shift_prior=Dirac(0.), scale_prior=Dirac(1.)),
            ],
        )
        @success (
            out(WarpedGaussianProcessParams([1.;1.;; 1.;1.;;], [1., 1.], [0.1, 0.1], [[0.0, 1.0], [0.0, 1.0]])) == 0.,
            out(WarpedGaussianProcessParams([1.;1.;; 1.;1.;;], [1., 1.], [0.1, 0.1], [[0.0, 1.0], [9.0, 1.0]])) == -Inf,
        )
    end
end

@testset "posterior: identity warp matches plain GaussianProcess" begin
    warped, plain = _twin_models(2; mean = x -> [1., 1.])
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = [2.;2.;; 5.;5.;; 8.;8.;;]
    data = ExperimentData(X, Y)

    w_params = WarpedGaussianProcessParams(
        [1.;1.;; 1.;1.;;], [1., 1.], [1e-4, 1e-4], [[0.0, 1.0], [0.0, 1.0]],
    )
    p_params = GaussianProcessParams([1.;1.;; 1.;1.;;], [1., 1.], [1e-4, 1e-4])

    w_post = model_posterior(warped, w_params, data)
    p_post = model_posterior(plain, p_params, data)

    Xtest = [1.;1.;; 3.;3.;; 100.;100.;;]
    @test isapprox(mean(w_post, Xtest), mean(p_post, Xtest); atol=1e-6)
    @test isapprox(var(w_post, Xtest), var(p_post, Xtest); atol=1e-6)
    @test isapprox(cov(w_post, Xtest), cov(p_post, Xtest); atol=1e-6)
    # median equals the mean for the identity warp (symmetric latent posterior)
    w_slice = model_posterior_slice(warped, w_params, data, 1)
    @test isapprox(BOSS.median(w_slice, [1., 1.]), mean(w_slice, [1., 1.]); atol=1e-6)
end

@testset "predictive_kind / predictive_samples" begin
    warped, _ = _twin_models(1; mean = x -> [1.])
    X = [2.;; 5.;; 8.;;]
    Y = [2.;; 5.;; 8.;;]
    data = ExperimentData(X, Y)
    w_params = WarpedGaussianProcessParams([1.;;], [1.], [1e-4], [[0.0, 1.0]])
    post = model_posterior_slice(warped, w_params, data, 1)

    @test predictive_kind(WarpedGaussianProcess) isa SampledPredictive
    @test predictive_kind(post) isa SampledPredictive

    x = [3.]
    ys, ws = predictive_samples(post, x)
    @test ys isa AbstractVector{<:Real}
    @test ws isa AbstractVector{<:Real}
    @test length(ys) == length(ws)
    @test isapprox(sum(ws), 1.; atol=1e-8)
    m, _ = mean_and_var(post, x)
    @test isapprox(sum(ws .* ys), m; atol=1e-6)

    Xtest = [1.;; 3.;; 100.;;]
    Ys, Ws = predictive_samples(post, Xtest)
    @test size(Ys) == size(Ws)
    @test all(isapprox.(vec(sum(Ws; dims=1)), 1.; atol=1e-8))
    mX, _ = mean_and_var(post, Xtest)
    @test isapprox(vec(sum(Ws .* Ys; dims=1)), mX; atol=1e-6)

    # a slice-native `SampledPredictive` model's *joint* posterior does not synthesize samples
    # by bundling its independent per-dimension slices (see `DefaultModelPosterior`)
    joint_post = model_posterior(warped, w_params, data)
    @test_throws MethodError predictive_samples(joint_post, x)
end

@testset "model_posterior(model, params, data)" begin
    warped, _ = _twin_models(2; mean = x -> [1., 1.])
    problem = BossProblem(;
        f = x -> x,
        domain = Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        acquisition = ExpectedImprovement(; fitness = LinFitness([1., 0.])),
        model = warped,
        data = ExperimentData([2.;2.;; 5.;5.;; 8.;8.;;], [2.;2.;; 5.;5.;; 8.;8.;;]),
    )
    BOSS.estimate_parameters!(problem, SamplingMAP(; samples=200, parallel=PARALLEL_TESTS); options=BossOptions(; info=false))

    @param_test model_posterior begin
        @params problem.model, problem.params, problem.data
        @success (
            out isa ModelPosterior,

            # types & shapes (vector query)
            mean(out, [2., 2.]) isa AbstractVector{<:Real},
            var(out, [2., 2.]) isa AbstractVector{<:Real},
            size(mean(out, [2., 2.])) == (2,),
            size(var(out, [2., 2.])) == (2,),
            mean_and_var(out, [2., 2.]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            isapprox(mean(out, [2., 2.]), mean_and_var(out, [2., 2.])[1]; atol=1e-8),
            isapprox(var(out, [2., 2.]), mean_and_var(out, [2., 2.])[2]; atol=1e-8),

            # recovery at training points (identity warp + tiny noise)
            isapprox(mean(out, [2., 2.]), [2., 2.]; atol=0.01),
            isapprox(mean(out, [5., 5.]), [5., 5.]; atol=0.01),
            isapprox(mean(out, [8., 8.]), [8., 8.]; atol=0.01),
            isapprox(mean(out, [100., 100.]), [1., 1.]; atol=0.01),  # reverts to mean fn
            all(var(out, [2., 2.]) .<= var(out, [3., 3.])),

            # types & shapes (matrix query)
            mean(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractMatrix{<:Real},
            var(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractMatrix{<:Real},
            cov(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractArray{<:Real, 3},
            size(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (2, 3),
            size(cov(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (3, 3, 2),
            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;])[:,2], mean(out, [2., 2.]); atol=1e-8),
            isapprox(var(out, [1.;1.;; 2.;2.;; 3.;3.;;])[:,3], var(out, [3., 3.]); atol=1e-8),
            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_cov(out, [1.;1.;; 2.;2.;; 3.;3.;;])[1]; atol=1e-8),
        )
    end
end

@testset "model_posterior_slice(model, params, data, slice)" begin
    warped, _ = _twin_models(2; mean = x -> [1., 1.])
    problem = BossProblem(;
        f = x -> x,
        domain = Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        acquisition = ExpectedImprovement(; fitness = LinFitness([1., 0.])),
        model = warped,
        data = ExperimentData([2.;2.;; 5.;5.;; 8.;8.;;], [2.;2.;; 5.;5.;; 8.;8.;;]),
    )
    BOSS.estimate_parameters!(problem, SamplingMAP(; samples=200, parallel=PARALLEL_TESTS); options=BossOptions(; info=false))

    @param_test model_posterior_slice begin
        @params problem.model, problem.params, problem.data, 1
        @params problem.model, problem.params, problem.data, 2
        @success (
            out isa ModelPosteriorSlice,

            # types & shapes (vector query)
            mean(out, [2., 2.]) isa Real,
            var(out, [2., 2.]) isa Real,
            std(out, [2., 2.]) isa Real,
            mean_and_var(out, [2., 2.]) isa Tuple{<:Real, <:Real},
            isapprox(mean(out, [2., 2.]), mean_and_var(out, [2., 2.])[1]; atol=1e-8),
            isapprox(std(out, [2., 2.]), sqrt(var(out, [2., 2.])); atol=1e-8),

            # recovery at training points
            isapprox(mean(out, [2., 2.]), 2.; atol=0.01),
            isapprox(mean(out, [8., 8.]), 8.; atol=0.01),
            isapprox(mean(out, [100., 100.]), 1.; atol=0.01),
            var(out, [2., 2.]) < var(out, [3., 3.]),

            # types & shapes (matrix query)
            mean(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractVector{<:Real},
            var(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractVector{<:Real},
            cov(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractMatrix{<:Real},
            size(cov(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (3, 3),
            mean_and_cov(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractMatrix{<:Real}},
            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;])[2], mean(out, [2., 2.]); atol=1e-8),
            isapprox(var(out, [1.;1.;; 2.;2.;; 3.;3.;;])[3], var(out, [3., 3.]); atol=1e-8),
            isapprox(cov(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_cov(out, [1.;1.;; 2.;2.;; 3.;3.;;])[2]; atol=1e-8),
        )
    end
end
