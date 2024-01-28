
@testset "random_start(bounds)" begin
    bounds = [0., 0.], [1., 1.]
    starts = [BOSS.random_start(bounds) for _ in 1:10]

    @test all((all(bounds[1] .<= s .<= bounds[2]) for s in starts))
    @test all((starts[i] != starts[i-1] for i in eachindex(starts)[2:end]))
end

@testset "generate_starts_LHC(bounds, count)" begin
    bounds = [0., 0.], [1., 1.]
    starts = BOSS.generate_starts_LHC(bounds, 10)

    @test all((all(bounds[1] .<= s .<= bounds[2]) for s in eachcol(starts)))
    @test all((starts[:,i] != starts[:,i-1] for i in 2:size(starts)[2]))
end
