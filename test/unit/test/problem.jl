
@testset "result(problem)" begin
    problem(X, Y) = BossProblem(;
        fitness = LinFitness([1., 0.]),
        f = x -> x,
        domain = Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        model = Nonparametric(;
            amp_priors = fill(BOSS.LogNormal(), 2),
            length_scale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
            noise_std_priors = fill(BOSS.Dirac(1e-4), 2),
        ),
        data = ExperimentDataPrior(X, Y),
    )

    @param_test result begin
        @params problem([2.;2.;; 5.;5.;; 8.;8.;;], [2.;2.;; 5.;5.;; 8.;8.;;])
        @success out == ([5., 5.], [5., 5.])

        @params problem([2.;2.;; 5.;5.;; 5.;5.;;], [2.;2.;; 5.;5.;; 8.;5.;;])
        @success out == ([5., 5.], [8., 5.])

        @params problem([2.;2.;; -1.;5.;; 8.;8.;;], [2.;2.;; 5.;5.;; 8.;8.;;])
        @success out == ([-1., 5.], [5., 5.])  # ignoring `domain` is correct behavior

        @params problem([2.;2.;; 5.;5.;; 8.;8.;;], [2.;10.;; 5.;10.;; 8.;10.;;])
        @success isnothing(out)
    end
end
