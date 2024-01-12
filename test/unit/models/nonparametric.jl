
@testset "Nonparametric" begin
    model = BOSS.Nonparametric(;
        mean = x -> [exp(x[1]), 0.],
        length_scale_priors = fill(BOSS.MvLogNormal([0.1], [1.]), 2),
    )
    data = BOSS.ExperimentDataMLE(
        [1.;; 2.;;],
        [1.;1.;; 4.;-1.;;],
        nothing,
        [1.;; 1.;;],
        [1e-8, 1e-8]
    )

    post = BOSS.model_posterior(model, data)
    @test post([1.])[1] ≈ [1., 1.] atol=1e-2
    @test post([2.])[1] ≈ [4., -1.] atol=1e-2
end
