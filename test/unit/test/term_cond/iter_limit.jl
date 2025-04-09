
@testset "(::IterLimit)(problem)" begin
    problem = BossProblem(;
        fitness = LinFitness([1.]),
        f = x -> [sin(x[1])],
        domain = Domain(; bounds=([0.], [10.])),
        model = Nonparametric(; amplitude_priors=[LogNormal()], lengthscale_priors=[BOSS.mvlognormal([1.], [1.])], noise_std_priors=[Dirac(0.1)]),
        data = ExperimentData(hcat([1.,2.,3.]...), hcat(sin.([1.,2.,3.])...)),
    )

    @param_test IterLimit begin
        @params 0
        @success (
            out(problem) == false,
            out(problem) == false,
        )

        @params 2
        @success (
            out(problem) == true,
            out(problem) == true,
            out(problem) == false,
            out(problem) == false,
        )
    end
end
