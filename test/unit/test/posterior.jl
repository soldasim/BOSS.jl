
@testset "average_mean(posteriors)" begin
    problem = BossProblem(;
        f = x -> x,
        domain = Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        acquisition = ExpectedImprovement(;
            fitness = LinFitness([1., 0.]),
        ),
        model = Nonparametric(;
            amplitude_priors = fill(LogNormal(), 2),
            lengthscale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
            noise_std_priors = fill(Dirac(1e-4), 2),
        ),
        data = ExperimentData([2.;2.;; 5.;5.;; 8.;8.;;], [2.;2.;; 5.;5.;; 8.;8.;;]),
    )
    turing = TuringBI(;
        sampler = PG(20),
        warmup = 20,
        samples_in_chain = 1,
        chain_count = 8,
        leap_size = 1,
        parallel = false,
    )
    BOSS.estimate_parameters!(problem, turing; options=BossOptions(; info=false))
    posts = model_posterior(problem)

    @test average_mean(posts, [3., 3.]) ≈ sum((mean(p, [3., 3.]) for p in posts)) / length(posts)
    @test average_mean(posts, [5., 5.]) ≈ sum((mean(p, [5., 5.]) for p in posts)) / length(posts)
end
