
@testset "construct_acquisition(acq, problem, options)" begin
    ei = ExpectedImprovement(;
        fitness = LinFitness([1.]),
    )

    problem = BossProblem(;
        f = x -> [sin(x[1])],
        domain = Domain(; bounds=([0.], [10.])),
        acquisition = ei,
        model = Nonparametric(; amplitude_priors=[BOSS.LogNormal()], lengthscale_priors=[BOSS.mvlognormal(ones(1), ones(1))], noise_std_priors=[BOSS.Dirac(0.1)]),
        data = ExperimentData(hcat([1.,2.,3.]...), hcat(sin.([1.,2.,3.])...)),
    )
    options = BossOptions(; info=false)
    BOSS.estimate_parameters!(problem, RandomFitter(); options)

    @param_test BOSS.construct_acquisition begin
        @params ei, problem, options
        @success (
            out isa Function,
            out([1.]) isa Real,
        )
    end
end

@testset "make_safe(acq, domain)" begin
    domain = Domain(; bounds = ([5., 5.], [10., 10.]))

    @param_test BOSS.make_safe begin 
        @params x -> x[1], domain
        @success (
            out isa Function,
            out([1.]) == 0.,
            out([5.]) == 5.,
            out([7.]) == 7.,
            out([10.]) == 10.,
            out([11.]) == 0.,
        )
    end
end

@testset "construct_ei(fitness, posterior, constraints, ϵ_samples, best_yet)" begin
    lin_fitness = LinFitness([1., 0.])
    nonlin_fitness = NonlinFitness(x -> x[1])
    posterior = ParametricPosterior(;
        f = identity,
        noise_std = [1., 1.],
    )
    posteriors = fill(posterior, 4)
    ϵ_samples = [
        0.422498  -1.33921    0.490985  -0.951167
        -0.289737   0.162767  -0.499742   0.892919
    ]

    @param_test BOSS.construct_ei begin
        @params lin_fitness, posterior, nothing, ϵ_samples, nothing
        @params nonlin_fitness, posterior, nothing, ϵ_samples, nothing
        @params lin_fitness, posteriors, nothing, ϵ_samples, nothing
        @params nonlin_fitness, posteriors, nothing, ϵ_samples, nothing
        @success (
            out isa Function,
            out([1., 1.]) == out([1., 1.]),
            out([1., 1.]) == 0.,
        )

        @params lin_fitness, posterior, [Inf, 10.], ϵ_samples, nothing
        @params nonlin_fitness, posterior, [Inf, 10.], ϵ_samples, nothing
        @params lin_fitness, posteriors, [Inf, 10.], ϵ_samples, nothing
        @params nonlin_fitness, posteriors, [Inf, 10.], ϵ_samples, nothing
        @success (
            out isa Function,
            out([1., 1.]) == out([1., 1.]),
            out([1., 1.]) > 0.,
            out([5., 1.]) == out([10., 1.]) == out([15., 1.]),
            out([1., 5.]) > out([1., 10.]) > out([1., 15.]),
            isapprox(out([1., 20.]), 0.; atol=1e-8),
        )

        @params lin_fitness, posterior, nothing, ϵ_samples, 10.
        @params nonlin_fitness, posterior, nothing, ϵ_samples, 10.
        @params lin_fitness, posteriors, nothing, ϵ_samples, 10.
        @params nonlin_fitness, posteriors, nothing, ϵ_samples, 10.
        @success (
            out isa Function,
            out([11., 1.]) == out([11., 1.]),
            out([11., 1.]) > 0.,
            out([5., 1.]) < out([10., 1.]) < out([15., 1.]),
            isapprox(out([0., 1.]), 0.; atol=1e-8),
            out([1., 5.]) == out([1., 10.]) == out([1., 15.]),
        )

        @params lin_fitness, posterior, [Inf, 10.], ϵ_samples, 10.
        @params nonlin_fitness, posterior, [Inf, 10.], ϵ_samples, 10.
        @params lin_fitness, posteriors, [Inf, 10.], ϵ_samples, 10.
        @params nonlin_fitness, posteriors, [Inf, 10.], ϵ_samples, 10.
        @success (
            out isa Function,
            out([11., 1.]) == out([11., 1.]),
            out([11., 1.]) > 0.,
            out([5., 1.]) < out([10., 1.]) < out([15., 1.]),
            out([11., 5.]) > out([11., 10.]) > out([11., 15.]),
            isapprox(out([1., 1.]), 0.; atol=1e-8),
            isapprox(out([11., 20.]), 0.; atol=1e-8),
        )
    end
end

@testset "expected_improvement(fitness, mean, std, ϵ_samples, best_yet)" begin
    lin_fitness = LinFitness([1., 0.])
    nonlin_fitness = NonlinFitness(x -> x[1])
    ϵ_samples = [
        0.422498  -1.33921    0.490985  -0.951167
        -0.289737   0.162767  -0.499742   0.892919
    ]
    ϵ = [0.422498, -0.289737]

    @param_test BOSS.expected_improvement begin
        @params lin_fitness, [0., 0.], [1., 1.], ϵ_samples, 0.
        @params lin_fitness, [0., 0.], [1., 1.], ϵ, 0.
        @params nonlin_fitness, [0., 0.], [1., 1.], ϵ_samples, 0.
        @params nonlin_fitness, [0., 0.], [1., 1.], ϵ, 0.
        @success out > 0.

        @params lin_fitness, [0., 0.], [0., 0.], ϵ_samples, 0.
        @params lin_fitness, [0., 0.], [0., 0.], ϵ, 0.
        @params nonlin_fitness, [0., 0.], [0., 0.], ϵ_samples, 0.
        @params nonlin_fitness, [0., 0.], [0., 0.], ϵ, 0.
        @success out == 0.

        @params lin_fitness, [1., 1.], [0., 0.], ϵ_samples, 0.
        @params lin_fitness, [1., 1.], [0., 0.], ϵ, 0.
        @params nonlin_fitness, [1., 1.], [0., 0.], ϵ_samples, 0.
        @params nonlin_fitness, [1., 1.], [0., 0.], ϵ, 0.
        @success out == 1.

        @params lin_fitness, [-10., -10.], [1., 1.], ϵ_samples, 0.
        @params lin_fitness, [-10., -10.], [1., 1.], ϵ, 0.
        @params nonlin_fitness, [-10., -10.], [1., 1.], ϵ_samples, 0.
        @params nonlin_fitness, [-10., -10.], [1., 1.], ϵ, 0.
        @success isapprox(out, 0.; atol=1e-20)
    end
end

@testset "feas_prob(mean, std, constraints)" begin
    @param_test BOSS.feas_prob begin
        @params [0., 0.], [0., 0.], nothing
        @success out == 1.
        @params [0., 0.], [1., 1.], nothing
        @success out == 1.
        @params [Inf, Inf], [1., 1.], nothing
        @success out == 1.

        @params [0., 0.], [1., 1.], [Inf, Inf]
        @success isapprox(out, 1.; atol=1e-20)
        @params [0., 0.], [1., 1.], [0., Inf]
        @success isapprox(out, 0.5; atol=1e-20)
        @params [0., 0.], [1., 1.], [0., 0.]
        @success isapprox(out, 0.25; atol=1e-20)

        @params [0., 0.], [1., 1.], [3., Inf]
        @success 0.99 < out < 1.
    end
end

@testset "best_so_far(fitness, X, Y, y_max, posterior)" begin
    @param_test BOSS.best_so_far begin
        @params LinFitness([1.]), [1.;; 2.;; 3.;;], [1.;; 2.;; 3.;;], [Inf]
        @params LinFitness([1.]), [1.;; 2.;; 3.;;], [1.;; 2.;; 3.;;], [5.]
        @params LinFitness([1.]), [10.;; 2.;; 3.;;], [10.;; 2.;; 3.;;], [5.]
        @success out == 3.

        @params LinFitness([2.]), [1.;; 2.;; 3.;;], [1.;; 2.;; 3.;;], [Inf]
        @success out == 6.

        @params LinFitness([1.]), [1.;; 2.;; 3.;;], [1.;; 2.;; 3.;;], [0.]
        @params LinFitness([1.]), Float64[;;], Float64[;;], [0.]
        @success isnothing(out)
    end
end
