
@testset "optimize_multistart" begin
    f(x) = (x, -sum(abs2, x .- 1.))
    flaky(x) = x[1] == 3. ? error("failed") : f(x)
    starts = [0. 3. 1.; 0. 3. 1.]
    options = BossOptions(; info=false)

    for (parallel, static_schedule) in ((false, false), (true, false), (true, true))
        kwargs = (; parallel, static_schedule, options)

        x, val = BOSS.optimize_multistart(f, starts; kwargs...)
        @test x == [1., 1.] && val == 0.

        xs, vals = BOSS.optimize_multistart(f, starts; return_all=true, kwargs...)
        @test length(xs) == 3 && length(vals) == 3

        # runs which throw are skipped
        x, val = BOSS.optimize_multistart(flaky, starts; kwargs...)
        @test x == [1., 1.] && val == 0.
        xs, vals = BOSS.optimize_multistart(flaky, starts; return_all=true, kwargs...)
        @test length(xs) == 2

        @test_throws ErrorException BOSS.optimize_multistart(x -> error("failed"), starts; kwargs...)
    end

    @test_throws ErrorException BOSS.optimize_multistart(flaky, starts; options=BossOptions(; info=false, debug=true))
end
