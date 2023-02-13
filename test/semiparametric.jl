
@testset "Semiparametric" begin
    model = BOSS.Semiparametric(
        BOSS.NonlinModel(;
            predict = (x,θ) -> [θ[1]*x[1] + θ[3], θ[2]*x[1] + θ[3]],
            param_priors = fill(Normal(), 3),
        ),
        BOSS.Nonparametric(;
            length_scale_priors = fill(MvLogNormal([0.1],Diagonal([1.])),2),
        )
    )
    data = BOSS.ExperimentDataMLE(
        [1.;; 2.;;],
        [3.;1.;; 6.;6.;;],
        [1.,-1.,2.],
        [1.;; 1.;;],
        [1e-8,1e-8]
    )

    post = BOSS.model_posterior(model, data)
    @test post([1.])[1] ≈ [3.,1.] atol=1e-2
    @test post([2.])[1] ≈ [6.,6.] atol=1e-2
end
