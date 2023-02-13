
@testset "LinModel" begin
    model = BOSS.LinModel(;
        lift = x -> [[sin(x[1]), 1.], [cos(x[2]), 1.]],
        param_priors = fill(Normal(), 4),
    )
    data = BOSS.ExperimentDataMLE(
        fill(0.,2,0),
        fill(0.,2,0),
        [0.5,1.2,-0.4,1.3],
        nothing,
        [0.1,0.1]
    )

    post = BOSS.model_posterior(model, data)
    mean, var = post([5.,6.])

    @test mean ≈ [0.5*sin(5.) + 1.2*1., -0.4*cos(6.) + 1.3*1.] atol=1e-8
    @test var ≈ [0.1, 0.1] atol=1e-8
end

@testset "NonlinModel" begin
    model = BOSS.NonlinModel(;
        predict = (x,θ) -> [θ[1] * exp(θ[2]*x[1]) * sin(θ[3]*x[1]), θ[4] * x[1] * x[2] + θ[5]],
        param_priors = fill(Normal(), 5),
    )
    data = BOSS.ExperimentDataMLE(
        fill(0.,2,0),
        fill(0.,2,0),
        [0.6,0.8,-0.2,-0.9,-1.0],
        nothing,
        [0.1,0.1]
    )

    post = BOSS.model_posterior(model, data)
    mean, var = post([6.3,4.9])
    @test mean ≈ [0.6 * exp(0.8*6.3) * sin(-0.2*6.3), -0.9*6.3*4.9 - 1.0] atol=1e-8
    @test var ≈ [0.1, 0.1] atol=1e-8
end
