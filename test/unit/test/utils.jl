
@testset "cond_func(f)" begin
    @param_test BOSS.cond_func begin
        @params x -> 0.
        @success (
            out(false, 1.) == 1.,
            out(true, 1.) == 0.,
        )
    end
end

@testset "in_domain(x, domain)" begin
    bounds = BOSS.Domain(;
        bounds = ([0., 0.], [10., 10.]),
    )
    discrete = BOSS.Domain(;
        bounds = ([0., 0.], [10., 10.]),
        discrete = [true, false],
    )
    cons = BOSS.Domain(;
        bounds = ([0., 0.], [10., 10.]),
        cons = x -> [5. - x[2]],  # x[2] <= 5.
    )
    complex = BOSS.Domain(
        bounds = ([0., 0.], [10., 10.]),
        discrete = [false, true],
        cons = x -> [5. - x[2]],  # x[2] <= 5.
    )


    @param_test BOSS.in_domain begin
        @params [0., 0.], bounds
        @params [5., 5.], bounds
        @params [10., 10.], bounds
        @params [0., 10.], bounds
        @success out == true

        @params [5., -1.], bounds
        @params [5., 11.], bounds
        @success out == false

        @params [0., 0.], discrete
        @params [5., 5.], discrete
        @params [5., 5.2], discrete
        @params [10., 10.], discrete
        @params [0., 10.], discrete
        @success out == true

        @params [5., -1.], discrete
        @params [5., 11.], discrete
        @params [-1., 5.], discrete
        @params [11., 5.], discrete
        @params [5.2, 5.2], discrete
        @params [-0.1, 5.], discrete
        @params [10.1, 5.], discrete
        @success out == false

        @params [0., 0.], cons
        @params [5., 3.], cons
        @params [5., 5.], cons
        @params [10., 5.], cons
        @params [10., 0.], cons
        @success out == true

        @params [5., -1.], cons
        @params [5., 11.], cons
        @params [5., 6.], cons
        @params [-1., 5.], cons
        @params [11., 5.], cons
        @params [11., 6.], cons
        @success out == false

        @params [0., 0.], complex
        @params [5., 3.], complex
        @params [5., 5.], complex
        @params [5.2, 5.], complex
        @params [10., 5.], complex
        @params [10., 0.], complex
        @success out == true

        @params [5., -1.], complex
        @params [5., 11.], complex
        @params [5., 6.], complex
        @params [-1., 5.], complex
        @params [11., 5.], complex
        @params [11., 6.], complex
        @params [5., 4.9], complex
        @params [-1., 4.9], complex
        @success out == false
    end
end

@testset "in_bounds(x, bounds)" begin
    @param_test BOSS.in_bounds begin
        @params [0.], ([0.], [10.])
        @params [5.], ([0.], [10.])
        @params [10.], ([0.], [10.])
        @params [0., 5.], ([0., 0.], [10., 10.])
        @params [5., 5.], ([0., 0.], [10., 10.])
        @params [10., 5.], ([0., 0.], [10., 10.])
        @params [0., 10.], ([0., 0.], [10., 10.])
        @success out == true

        @params [-1.], ([0.], [10.])
        @params [10.1], ([0.], [10.])
        @params [-1., 5.], ([0., 0.], [10., 10.])
        @params [10.1, 5.], ([0., 0.], [10., 10.])
        @params [-1., 11.], ([0., 0.], [10., 10.])
        @success out == false
    end
end

@testset "in_discrete(x, discrete)" begin
    @param_test BOSS.in_discrete begin
        @params [1.], [true]
        @params [1.], [false]
        @params [1.2], [false]
        @params [1., 1.], [true, true]
        @params [1., 1.], [false, true]
        @params [1.2, 1.], [false, true]
        @success out == true

        @params [1.2], [true]
        @params [1., 1.2], [false, true]
        @params [1.2, 1.2], [false, true]
        @success out == false
    end
end

@testset "in_cons(x, cons)" begin
    @param_test BOSS.in_cons begin
        @params [1.], nothing
        @params [1., 1.], nothing
        @success out == true

        @params [1.], x -> x
        @params [0.], x -> x
        @params [1., 1.], x -> [x[1] + x[2]]
        @params [1., -1.], x -> [x[1] + x[2]]
        @success out == true

        @params [-1.], x -> x
        @params [0., -1.], x -> [x[1] + x[2]]
        @success out == false
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

@testset "result(problem)" begin
    problem(X, Y) = BOSS.BossProblem(;
        fitness = BOSS.LinFitness([1., 0.]),
        f = x -> x,
        domain = BOSS.Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        model = BOSS.Nonparametric(;
            amp_priors = fill(BOSS.LogNormal(), 2),
            length_scale_priors = fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2),
        ),
        noise_var_priors = fill(BOSS.Dirac(1e-8), 2),
        data = BOSS.ExperimentDataPrior(X, Y),
    )

    @param_test result begin
        @params problem([2.;2.;; 5.;5.;; 8.;8.;;], [2.;2.;; 5.;5.;; 8.;8.;;])
        @success out == ([5., 5.], [5., 5.])

        @params problem([2.;2.;; 5.;5.;; 5.;5.;;], [2.;2.;; 5.;5.;; 8.;5.;;])
        @success out == ([5., 5.], [8., 5.])

        @params problem([2.;2.;; -1.;5.;; 8.;8.;;], [2.;2.;; 5.;5.;; 8.;8.;;])
        @success out == ([-1., 5.], [5., 5.])  # ignoring `bounds` is correct behavior

        @params problem([2.;2.;; 5.;5.;; 8.;8.;;], [2.;10.;; 5.;10.;; 8.;10.;;])
        @success isnothing(out)
    end
end

@testset "exclude_exterior_points(domain, X, Y)" begin
    domain = BOSS.Domain(;
        bounds = ([0., 0.], [10., 10.]),
        discrete = [true, false],
        cons = x -> [x[2] - 5.],  # x[2] >= 5.
    )
    options = BOSS.BossOptions(; info=false)

    @param_test (in...) -> BOSS.exclude_exterior_points(in...; options) begin
        @params deepcopy(domain), [5.;5.;; 5.;5.;; 8.;8.;;], [5.;5.;; 5.;5.;; 8.;8.;;]
        @success out == (in[2], in[3])

        @params deepcopy(domain), [2.;2.;; 5.;5.;; 8.;8.;;], [2.;2.;; 5.;5.;; 8.;8.;;]
        @params deepcopy(domain), [5.2;5.2;; 5.;5.;; 8.;8.;;], [5.2;5.2;; 5.;5.;; 8.;8.;;]
        @params deepcopy(domain), [11.;11.;; 5.;5.;; 8.;8.;;], [11.;11.;; 5.;5.;; 8.;8.;;]
        @success out == ([5.;5.;; 8.;8.;;], [5.;5.;; 8.;8.;;])
    end
end
