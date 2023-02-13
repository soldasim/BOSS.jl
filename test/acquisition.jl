
@testset "Feasibility probabilty" begin
    mean = [5.,5.]
    var = [1.,1.]
    constraints = [4.,6.]

    @test BOSS.feas_prob(mean, var, constraints) ≈ 0.1334837643314019 atol=1e-8
end

@testset "EI with LinFitness" begin
    fitness = BOSS.LinFitness([2.,1.])
    mean = [5.,10.]
    var = [1.,1.]
    best_yet = 8.

    @test BOSS.EI(mean, var, fitness; best_yet) ≈ 12.000000015720124 atol=1e-8
end

@testset "EI with NonlinFitness" begin
    fitness = BOSS.NonlinFitness(y -> cos(y[1]) * sin(y[2]))
    mean = [5.,7.]
    var = [0.1,0.1]
    best_yet = 0.05

    ϵ = [0.5745583838742907, -1.3177536848614708]
    ϵ_samples = [
        -0.571173  -0.797771  -0.293224   -0.421724
        -0.305551  -0.732456   0.0428198   0.66368
    ]

    @test BOSS.EI(mean, var, fitness, ϵ; best_yet) ≈ 0.13679782913530203 atol=1e-8
    @test BOSS.EI(mean, var, fitness, ϵ_samples; best_yet) ≈ 0.10216139253430176 atol=1e-8
end
