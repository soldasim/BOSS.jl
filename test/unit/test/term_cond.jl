
@testset "(::IterLimit)(problem)" begin
    problem = BOSS.BossProblem(;
        fitness = BOSS.LinFitness([1.]),
        f = x -> [sin(x[1])],
        domain = BOSS.Domain(; bounds=([0.], [10.])),
        model = BOSS.Nonparametric(; length_scale_priors=[BOSS.MvLogNormal([1.], [1.])]),
        noise_var_priors = [BOSS.Dirac(0.1)],
        data = BOSS.ExperimentDataPrior(hcat([1.,2.,3.]...), hcat(sin.([1.,2.,3.])...)),
    )

    @param_test BOSS.IterLimit begin
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
