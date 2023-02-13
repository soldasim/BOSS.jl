
dummy_problem() = BOSS.OptimizationProblem(;
    fitness = BOSS.LinFitness([1.]),
    f = x->[sin(x[1])],
    cons = [Inf],
    domain = BOSS.OptimDomain(nothing),
    discrete = [false],
    model = BOSS.Nonparametric(; length_scale_priors=[MvLogNormal([1.],Diagonal([1.]))]),
    noise_var_priors = [LogNormal(-2.3, 1.)],
    data = BOSS.ExperimentDataPrior(hcat(1.,2.,3.), hcat(sin(1.),sin(2.),sin(3.))),
)

@testset "IterLimit" begin
    problem = dummy_problem()
    cond = BOSS.IterLimit(2)

    @test cond(problem) == true
    @test cond(problem) == true
    @test cond(problem) == false
end
