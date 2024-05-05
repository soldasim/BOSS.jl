
@testset "(::DiscreteKernel)(x1, x2)" begin
    @param_test BOSS.DiscreteKernel begin
        @params BOSS.Matern52Kernel(), [false, false]
        @success (
            out([1.2, 1.2], [3.8, 3.8]) == BOSS.Matern52Kernel()([1.2, 1.2], [3.8, 3.8]),
            out([1.2, 2.], [3.8, 4.]) == BOSS.Matern52Kernel()([1.2, 2.], [3.8, 4.]),
        )

        @params BOSS.Matern52Kernel(), [false, true]
        @success (
            out([1.2, 1.2], [3.8, 3.8]) == out([1.2, 1.], [3.8, 4.]),
            out([1.2, 1.2], [3.8, 4.]) == out([1.2, 1.], [3.8, 4.]),
        )
    end
end

@testset "make_discrete(model, discrete)" begin
    @param_test BOSS.make_discrete begin
        @params (
            BOSS.Nonparametric(;
                mean = x -> [x[1], 0.],
                kernel = BOSS.Matern52Kernel(),
                amp_priors = fill(BOSS.LogNormal(), 2),
                length_scale_priors = fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2),
            ),
            [false, true],
        )
        @success (
            out.kernel isa BOSS.DiscreteKernel,
            out.kernel([1.2, 1.2], [3.8, 3.8]) == out.kernel([1.2, 1.], [3.8, 4.]),
            out.kernel([1.2, 1.2], [3.8, 3.8]) != in[1].kernel([1.2, 1.2], [3.8, 3.8]),
        )

        @params (
            BOSS.Nonparametric(;
                mean = x -> [x[1], 0.],
                kernel = BOSS.DiscreteKernel(BOSS.Matern52Kernel(), [false, true]),
                amp_priors = fill(BOSS.LogNormal(), 2),
                length_scale_priors = fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2),
            ),
            [false, true],
        )
        @success (
            out.kernel isa BOSS.DiscreteKernel,
            out.kernel([1.2, 1.2], [3.8, 3.8]) == out.kernel([1.2, 1.], [3.8, 4.]),
            out.kernel([1.2, 1.2], [3.8, 3.8]) == in[1].kernel([1.2, 1.2], [3.8, 3.8]),
        )

        @params (
            BOSS.Nonparametric(;
                mean = x -> [x[1], 0.],
                kernel = BOSS.Matern52Kernel(),
                amp_priors = fill(BOSS.LogNormal(), 2),
                length_scale_priors = fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2),
            ),
            [false, false],
        )
        @success (
            out.kernel isa BOSS.DiscreteKernel,
            out.kernel([1.2, 1.2], [3.8, 3.8]) != out.kernel([1., 1.], [4., 4.]),
            out.kernel([1.2, 1.2], [3.8, 3.8]) == in[1].kernel([1.2, 1.2], [3.8, 3.8]),
        )
    end
end

@testset "make_discrete(kernel, discrete)" begin
    @param_test BOSS.make_discrete begin
        @params BOSS.Matern52Kernel(), [false, false]
        @success (
            out isa BOSS.DiscreteKernel,
            out.kernel == in[1],
            out([1.2, 1.2], [3.8, 3.8]) != out([1.2, 1.], [3.8, 4.]),
            out([1.2, 1.2], [3.8, 3.8]) == in[1]([1.2, 1.2], [3.8, 3.8]),
        )

        @params BOSS.Matern52Kernel(), [false, true]
        @success (
            out isa BOSS.DiscreteKernel,
            out.kernel == in[1],
            out([1.2, 1.2], [3.8, 3.8]) == out([1.2, 1.], [3.8, 4.]),
            out([1.2, 1.2], [3.8, 3.8]) != in[1]([1.2, 1.2], [3.8, 3.8]),
        )
    end
end

@testset "model_posterior(model, data)" begin
    problem = BOSS.BossProblem(;
        fitness = BOSS.LinFitness([1., 0.]),
        f = x -> x,
        domain = BOSS.Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        model = BOSS.Nonparametric(;
            mean = x -> [1., 1.],
            amp_priors = fill(BOSS.LogNormal(), 2),
            length_scale_priors = fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2),
        ),
        noise_var_priors = fill(BOSS.Dirac(1e-8), 2),
        data = BOSS.ExperimentDataPrior([2.;2.;; 5.;5.;; 8.;8.;;], [2.;2.;; 5.;5.;; 8.;8.;;]),
    )
    BOSS.estimate_parameters!(problem, BOSS.SamplingMLE(; samples=200, parallel=PARALLEL_TESTS); options=BOSS.BossOptions(; info=false))

    @param_test (m, d) -> BOSS.model_posterior(m, d; split=false) begin
        @params problem.model, problem.data
        @success (
            out isa Function,
            out([2., 2.]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            length(out([2., 2.])[1]) == 2,
            length(out([2., 2.])[2]) == 2,
            isapprox.(out([2., 2.])[1], [2., 2.]; atol=0.01) |> all,
            isapprox.(out([5., 5.])[1], [5., 5.]; atol=0.01) |> all,
            isapprox.(out([8., 8.])[1], [8., 8.]; atol=0.01) |> all,
            all(out([1., 1.])[1] .< [2., 2.]),
            all(out([4., 4.])[1] .< [5., 5.]),
            isapprox.(out([100., 100.])[1], [1., 1.]; atol=0.01) |> all,
            all(out([2., 2.])[2] .< out([3., 3.])[2]),
            all(out([10., 10.])[2] .< out([11., 11.])[2]),
        )
    end
    @param_test (m, d) -> BOSS.model_posterior(m, d; split=true) begin
        @params problem.model, problem.data
        @success (
            out isa Vector,
            out[1]([2., 2.]) isa Tuple{<:Real, <:Real},
            out[2]([2., 2.]) isa Tuple{<:Real, <:Real},
            isapprox(out[1]([2., 2.])[1], 2.; atol=0.01),
            isapprox(out[2]([2., 2.])[1], 2.; atol=0.01),
            isapprox(out[1]([5., 5.])[1], 5.; atol=0.01),
            isapprox(out[2]([5., 5.])[1], 5.; atol=0.01),
            isapprox(out[1]([8., 8.])[1], 8.; atol=0.01),
            isapprox(out[2]([8., 8.])[1], 8.; atol=0.01),
            out[1]([1., 1.])[1] < 2.,
            out[2]([1., 1.])[1] < 2.,
            out[1]([4., 4.])[1] < 5.,
            out[2]([4., 4.])[1] < 5.,
            isapprox(out[1]([100., 100.])[1], 1.; atol=0.01),
            isapprox(out[2]([100., 100.])[1], 1.; atol=0.01),
            out[1]([2., 2.])[2] < out[1]([3., 3.])[2],
            out[2]([2., 2.])[2] < out[2]([3., 3.])[2],
            out[1]([10., 10.])[2] < out[1]([11., 11.])[2],
            out[2]([10., 10.])[2] < out[2]([11., 11.])[2],
        )
    end
end

@testset "model_loglike(model, noise_var_priors, data)" begin
    model = BOSS.Nonparametric(;
        mean = x -> [1., 1.],
        amp_priors = fill(BOSS.LogNormal(), 2),
        length_scale_priors = fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2),
    )
    noise_var_priors = fill(BOSS.LogNormal(), 2)
    data = BOSS.ExperimentDataPrior([2.;2.;; 5.;5.;; 8.;8.;;], [2.;2.;; 5.;5.;; 8.;8.;;])

    @param_test BOSS.model_loglike begin
        @params deepcopy(model), deepcopy(noise_var_priors), deepcopy(data)
        @success (
            out isa Function,
            out(Float64[], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.]) isa Real,
            out(Float64[], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.]) < 0.,
            out(Float64[], [1.;1.;; 1.;1.;;], [1., 1.], [5., 5.]) > out(Float64[], [1.;1.;; 1.;1.;;], [1., 1.], [100., 100.]),
            out(Float64[], [1.;1.;; 1.;1.;;], [5., 5.], [1., 1.]) > out(Float64[], [1.;1.;; 1.;1.;;], [100., 100.], [1., 1.]),
            out(Float64[], [5.;5.;; 5.;5.;;], [1., 1.], [1., 1.]) > out(Float64[], [100.;100.;; 100.;100.;;], [1., 1.], [1., 1.]),
        )
    end
end

@testset "model_params_loglike(model, λ, α)" begin
    @param_test BOSS.model_params_loglike begin
        # TODO: Add different priors loaded from a collection.
        @params (
            BOSS.Nonparametric(;
                amp_priors = fill(BOSS.LogNormal(), 2),
                length_scale_priors = fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2),
            ),
            [1.;1.;; 5.;5.;;],
            [1., 2.],
        )
        @success isapprox(
            out,
            sum((BOSS.logpdf(in[1].length_scale_priors[i], in[2][:,i]) for i in 1:2))
                + sum((BOSS.logpdf(in[1].amp_priors[i], in[3][i]) for i in 1:2));
            atol=1e-20,
        )

        @params (
            BOSS.Nonparametric(;
                amp_priors = fill(BOSS.Dirac(1.), 2),
                length_scale_priors = fill(BOSS.Product(fill(BOSS.Dirac(1.), 2)), 2),
            ),
            [1.;1.;; 1.;1.;;],
            [1., 1.],
        )
        @success out == 0.

        @params (
            BOSS.Nonparametric(;
                amp_priors = fill(BOSS.Dirac(1.), 2),
                length_scale_priors = fill(BOSS.Product(fill(BOSS.Dirac(1.), 2)), 2),
            ),
            [1.;1.;; 5.;5.;;],
            [1., 1.],
        )
        @success out == -Inf
    end
end

# TODO: Check the GP marginal data likelihood against the equiation in
# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7352306 section III. D.
@testset "model_data_loglike(model, λ, α, noise_vars, X, Y)" begin
    model = BOSS.Nonparametric(;
        mean = x -> [0.],
        amp_priors = fill(BOSS.LogNormal(), 1),
        length_scale_priors = fill(BOSS.Product(fill(BOSS.Dirac(1.), 1)), 1),
    )

    @param_test BOSS.model_data_loglike begin
        @params deepcopy(model), [1.;;], [1.], [1.], [1.;; 2.;; 3.;;], [1.;; 2.;; 3.;;]
        @success out < 0.
    end

    t_length_scale(λ) = BOSS.model_data_loglike(deepcopy(model), λ, [1.], [0.], [1.;; 2.;; 3.;;], [1.;; -1.;; 1.;;])
    @test t_length_scale([0.1;;]) > t_length_scale([1.;;]) > t_length_scale([10.;;])

    t_amplitude(α) = BOSS.model_data_loglike(deepcopy(model), [1.;;], α, [0.], [1.;; 2.;; 3.;;], [1.;; -1.;; 1.;;])
    @test t_amplitude([1.]) > t_amplitude([0.1])
    @test t_amplitude([1.]) > t_amplitude([10.])

    t_noise_var(σ2) = BOSS.model_data_loglike(deepcopy(model), [1.;;], [1.], σ2, [1.;; 2.;; 3.;;], [1.;; -1.;; 1.;;])
    @test t_noise_var([1.]) > t_noise_var([0.1])
    @test t_noise_var([1.]) > t_noise_var([10.])

    t_data(X, Y) = BOSS.model_data_loglike(deepcopy(model), [1.;;], [1.], [0.], X, Y)
    @test t_data([1.;; 2.;; 3;;], [99.9;; 100.;; 100.1;;]) > t_data([1.;; 2.;; 3;;], [99.;; 100.;; 101.;;]) > t_data([1.;; 2.;; 3;;], [90.;; 100.;; 110.;;])
end
