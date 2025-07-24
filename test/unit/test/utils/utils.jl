
@testset "ith(i)" begin
    v = [1,2,3]
    vs = [[11,12,13],[21,22,23],[31,32,33]]

    @test BOSS.ith(2)(v) == 2
    @test BOSS.ith(2).(vs) == [12,22,32]
end

@testset "cond_func(f)" begin
    f(x) = 0.

    @test BOSS.cond_func(f)(5., false) == 5.
    @test BOSS.cond_func(f)(5., true) == 0.
    @test BOSS.cond_func(f).([5., 5.], [false, true]) == [5., 0.]
end

@testset "discrete_round(dims, x)" begin
    @param_test BOSS.discrete_round begin
        @params nothing, [4.2, 5.3]
        @params [false, false], [4.2, 5.3]
        @success out == [4.2, 5.3]

        @params missing, [4.2, 5.3]
        @params [true, true], [4.2, 5.3]
        @success out == [4., 5.]

        @params [true, false], [4.2, 5.3]
        @success out == [4., 5.3]
    end
end

@testset "is_feasible(y, y_max)" begin
    @param_test BOSS.is_feasible begin
        @params [1.], [Inf]
        @params [Inf], [Inf]
        @params [1.], [10.]
        @params [10.], [10.]
        @params [1., 11.], [10., Inf]
        @params [10., 11.], [10., Inf]
        @params [-Inf, 1.], [-Inf, 10.]
        @success out == true

        @params [11.], [10.]
        @params [1., 11.], [10., 10.]
        @params [11., 11.], [10., 10.]
        @params [Inf], [10.]
        @params [Inf], [-Inf]
        @params [0., 1.], [-Inf, 10.]
        @success out == false
    end
end

@testset "random_point(bounds)" begin
    bounds = [0., 0.], [1., 1.]
    starts = [BOSS.random_point(bounds) for _ in 1:10]

    @test all((all(bounds[1] .<= s .<= bounds[2]) for s in starts))
    @test all((starts[i] != starts[i-1] for i in eachindex(starts)[2:end]))
end

@testset "generate_LHC(bounds, count)" begin
    bounds = [0., 0.], [1., 1.]
    starts = BOSS.generate_LHC(bounds, 10)

    @test all((all(bounds[1] .<= s .<= bounds[2]) for s in eachcol(starts)))
    @test all((starts[:,i] != starts[:,i-1] for i in 2:size(starts)[2]))
end

@testset "result(problem)" begin
    problem(X, Y) = BossProblem(;
        f = x -> x,
        domain = Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        model = Nonparametric(;
            amplitude_priors = fill(LogNormal(), 2),
            lengthscale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
            noise_std_priors = fill(Dirac(1e-4), 2),
        ),
        acquisition = ExpectedImprovement(;
            fitness = LinFitness([1, 0]),
        ),
        data = ExperimentData(X, Y),
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
