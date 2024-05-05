
@testset "average_posteriors(posteriors)" begin
    problem = BOSS.BossProblem(;
        fitness = BOSS.LinFitness([1., 0.]),
        f = x -> x,
        domain = BOSS.Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        model = BOSS.Nonparametric(;
            amp_priors = fill(BOSS.LogNormal(), 2),
            length_scale_priors = fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2),
        ),
        noise_var_priors = fill(BOSS.Dirac(1e-8), 2),
        data = BOSS.ExperimentDataPrior([2.;2.;; 5.;5.;; 8.;8.;;], [2.;2.;; 5.;5.;; 8.;8.;;]),
    )
    turing = BOSS.TuringBI(;
        sampler = BOSS.PG(20),
        warmup = 20,
        samples_in_chain = 1,
        chain_count = 8,
        leap_size = 1,
        parallel = false,
    )
    BOSS.estimate_parameters!(problem, turing; options=BOSS.BossOptions(; info=false))
    posteriors = BOSS.model_posterior(problem.model, problem.data)

    @param_test BOSS.average_posteriors begin
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

@testset "discrete_round(dims, x)" begin
    @param_test BOSS.discrete_round begin
        @params nothing, [4.2, 5.3]
        @params [false, false], [4.2, 5.3]
        @success out == [4.2, 5.3]

        @params missing, [4.2, 5.3]
        @params [true, true], [4.2, 5.3]
        @success out == [4., 5.]

        @params [true, false], [4.2, 5.3]
        @success out == [4., 5.3]
    end
end

noise_loglike(noise_var_priors, noise_vars) = mapreduce(p -> logpdf(p...), +, zip(noise_var_priors, noise_vars))

@testset "noise_loglike(noise_var_priors, noise_vars)" begin
    @param_test BOSS.noise_loglike begin
        # TODO: Add different priors loaded from a collection.
        @params fill(BOSS.LogNormal(), 2), [0.1, 0.1]
        @success out == 2 * BOSS.logpdf(BOSS.LogNormal(), 0.1)

        @params [BOSS.LogNormal(), BOSS.Dirac(0.1)], [0.1, 0.1]
        @success out == BOSS.logpdf(BOSS.LogNormal(), 0.1)

        @params fill(BOSS.Dirac(0.1), 2), [0.1, 0.1]
        @success out == 0.

        @params fill(BOSS.Dirac(0.1), 2), [0.1, 1.]
        @success out == -Inf
    end
end
