
@testset "average_posterior(posteriors)" begin
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
    posteriors = model_posterior(problem)

    @param_test average_posterior begin
        @params posteriors
        @success (
            out isa Function,
            isapprox(out([3., 3.])[1], sum((p([3., 3.])[1] for p in posteriors)) / length(posteriors); atol=1e-20),
            isapprox(out([3., 3.])[2], sum((p([3., 3.])[2] for p in posteriors)) / length(posteriors); atol=1e-20),
            isapprox(out([5., 5.])[1], sum((p([5., 5.])[1] for p in posteriors)) / length(posteriors); atol=1e-20),
            isapprox(out([5., 5.])[2], sum((p([5., 5.])[2] for p in posteriors)) / length(posteriors); atol=1e-20),
        )
    end
end
