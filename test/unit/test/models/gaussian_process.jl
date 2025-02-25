
@testset "(::DiscreteKernel)(x1, x2)" begin
    @param_test BOSS.DiscreteKernel begin
        @params Matern32Kernel(), [false, false]
        @success (
            out([1.2, 1.2], [3.8, 3.8]) == Matern32Kernel()([1.2, 1.2], [3.8, 3.8]),
            out([1.2, 2.], [3.8, 4.]) == Matern32Kernel()([1.2, 2.], [3.8, 4.]),
        )

        @params Matern32Kernel(), [false, true]
        @success (
            out([1.2, 1.2], [3.8, 3.8]) == out([1.2, 1.], [3.8, 4.]),
            out([1.2, 1.2], [3.8, 4.]) == out([1.2, 1.], [3.8, 4.]),
        )
    end
end

@testset "make_discrete(model, discrete)" begin
    @param_test BOSS.make_discrete begin
        @params (
            Nonparametric(;
                mean = x -> [x[1], 0.],
                kernel = Matern32Kernel(),
                amp_priors = fill(LogNormal(), 2),
                length_scale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
                noise_std_priors = fill(Dirac(0.1), 2),
            ),
            [false, true],
        )
        @success (
            out.kernel isa BOSS.DiscreteKernel,
            out.kernel([1.2, 1.2], [3.8, 3.8]) == out.kernel([1.2, 1.], [3.8, 4.]),
            out.kernel([1.2, 1.2], [3.8, 3.8]) != in[1].kernel([1.2, 1.2], [3.8, 3.8]),
        )

        @params (
            Nonparametric(;
                mean = x -> [x[1], 0.],
                kernel = BOSS.DiscreteKernel(Matern32Kernel(), [false, true]),
                amp_priors = fill(LogNormal(), 2),
                length_scale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
                noise_std_priors = fill(Dirac(0.1), 2),
            ),
            [false, true],
        )
        @success (
            out.kernel isa BOSS.DiscreteKernel,
            out.kernel([1.2, 1.2], [3.8, 3.8]) == out.kernel([1.2, 1.], [3.8, 4.]),
            out.kernel([1.2, 1.2], [3.8, 3.8]) == in[1].kernel([1.2, 1.2], [3.8, 3.8]),
        )

        @params (
            Nonparametric(;
                mean = x -> [x[1], 0.],
                kernel = Matern32Kernel(),
                amp_priors = fill(LogNormal(), 2),
                length_scale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
                noise_std_priors = fill(Dirac(0.1), 2),
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
        @params Matern32Kernel(), [false, false]
        @success (
            out isa BOSS.DiscreteKernel,
            out.kernel == in[1],
            out([1.2, 1.2], [3.8, 3.8]) != out([1.2, 1.], [3.8, 4.]),
            out([1.2, 1.2], [3.8, 3.8]) == in[1]([1.2, 1.2], [3.8, 3.8]),
        )

        @params Matern32Kernel(), [false, true]
        @success (
            out isa BOSS.DiscreteKernel,
            out.kernel == in[1],
            out([1.2, 1.2], [3.8, 3.8]) == out([1.2, 1.], [3.8, 4.]),
            out([1.2, 1.2], [3.8, 3.8]) != in[1]([1.2, 1.2], [3.8, 3.8]),
        )
    end
end

@testset "model_posterior(model, data)" begin
    problem = BossProblem(;
        fitness = LinFitness([1., 0.]),
        f = x -> x,
        domain = Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        model = Nonparametric(;
            mean = x -> [1., 1.],
            amp_priors = fill(LogNormal(), 2),
            length_scale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
            noise_std_priors = fill(Dirac(1e-4), 2),
        ),
        data = ExperimentDataPrior([2.;2.;; 5.;5.;; 8.;8.;;], [2.;2.;; 5.;5.;; 8.;8.;;]),
    )
    BOSS.estimate_parameters!(problem, SamplingMAP(; samples=200, parallel=PARALLEL_TESTS); options=BossOptions(; info=false))

    @param_test model_posterior begin
        @params problem.model, problem.data
        @success (
            out isa Function,

            # vector
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
            
            # matrix
            out([1.;1.;; 2.;2.;; 3.;3.;;]) isa Tuple{<:AbstractMatrix{<:Real}, <:AbstractArray{<:Real, 3}},
            size(out([1.;1.;; 2.;2.;; 3.;3.;;])[1]) == (3, 2),
            size(out([1.;1.;; 2.;2.;; 3.;3.;;])[2]) == (3, 3, 2),
            isapprox.(out([1.;1.;; 2.;2.;; 3.;3.;;])[1][1,:], out([1., 1.])[1]; atol=1e-8) |> all,
            isapprox.(out([1.;1.;; 2.;2.;; 3.;3.;;])[1][2,:], out([2., 2.])[1]; atol=1e-8) |> all,
            isapprox.(out([1.;1.;; 2.;2.;; 3.;3.;;])[1][3,:], out([3., 3.])[1]; atol=1e-8) |> all,
            isapprox.(out([1.;1.;; 2.;2.;; 3.;3.;;])[2][1,1,:], out([1., 1.])[2] .^ 2; atol=1e-8) |> all,
            isapprox.(out([1.;1.;; 2.;2.;; 3.;3.;;])[2][2,2,:], out([2., 2.])[2] .^ 2; atol=1e-8) |> all,
            isapprox.(out([1.;1.;; 2.;2.;; 3.;3.;;])[2][3,3,:], out([3., 3.])[2] .^ 2; atol=1e-8) |> all,

            # single-element matrix
            out([1.;1.;;]) isa Tuple{<:AbstractMatrix{<:Real}, <:AbstractArray{<:Real, 3}},
            size(out([1.;1.;;])[1]) == (1, 2),
            size(out([1.;1.;;])[2]) == (1, 1, 2),
        )
    end
end

@testset "model_posterior_slice(model, data, slice)" begin
    problem = BossProblem(;
        fitness = LinFitness([1., 0.]),
        f = x -> x,
        domain = Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        model = Nonparametric(;
            mean = x -> [1., 1.],
            amp_priors = fill(LogNormal(), 2),
            length_scale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
            noise_std_priors = fill(Dirac(1e-4), 2),
        ),
        data = ExperimentDataPrior([2.;2.;; 5.;5.;; 8.;8.;;], [2.;2.;; 5.;5.;; 8.;8.;;]),
    )
    BOSS.estimate_parameters!(problem, SamplingMAP(; samples=200, parallel=PARALLEL_TESTS); options=BossOptions(; info=false))

    @param_test model_posterior_slice begin
        @params problem.model, problem.data, 1
        @params problem.model, problem.data, 2
        @success (
            out isa Function,

            # vector
            out([2., 2.]) isa Tuple{<:Real, <:Real},
            isapprox(out([2., 2.])[1], 2.; atol=0.01),
            isapprox(out([5., 5.])[1], 5.; atol=0.01),
            isapprox(out([8., 8.])[1], 8.; atol=0.01),
            out([1., 1.])[1] < 2.,
            out([4., 4.])[1] < 5.,
            isapprox(out([100., 100.])[1], 1.; atol=0.01),
            out([2., 2.])[2] < out([3., 3.])[2],
            out([10., 10.])[2] .< out([11., 11.])[2],

            # matrix
            out([1.;1.;; 2.;2.;; 3.;3.;;]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractMatrix{<:Real}},
            size(out([1.;1.;; 2.;2.;; 3.;3.;;])[1]) == (3,),
            size(out([1.;1.;; 2.;2.;; 3.;3.;;])[2]) == (3, 3),
            isapprox(out([1.;1.;; 2.;2.;; 3.;3.;;])[1][1], out([1., 1.])[1]; atol=1e-8),
            isapprox(out([1.;1.;; 2.;2.;; 3.;3.;;])[1][2], out([2., 2.])[1]; atol=1e-8),
            isapprox(out([1.;1.;; 2.;2.;; 3.;3.;;])[1][3], out([3., 3.])[1]; atol=1e-8),
            isapprox(out([1.;1.;; 2.;2.;; 3.;3.;;])[2][1,1], out([1., 1.])[2] ^ 2; atol=1e-8),
            isapprox(out([1.;1.;; 2.;2.;; 3.;3.;;])[2][2,2], out([2., 2.])[2] ^ 2; atol=1e-8),
            isapprox(out([1.;1.;; 2.;2.;; 3.;3.;;])[2][3,3], out([3., 3.])[2] ^ 2; atol=1e-8),

            # single-element matrix
            out([1.;1.;;]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractMatrix{<:Real}},
            size(out([1.;1.;;])[1]) == (1,),
            size(out([1.;1.;;])[2]) == (1, 1),
        )
    end
end

@testset "_clip_var(var)" begin
    max_neg_var = 1e-8

    @param_test var -> BOSS._clip_var(var; treshold=max_neg_var) begin
        @params 0.
        @params 1e-9
        @params 1e-8
        @params 1e-7
        @params 1.
        @success out == in[1]

        @params -1e-9
        @params -1e-8
        @success out == 0.

        @params -1e-7
        @failure DomainError
    end
end

@testset "model_loglike(model, data)" begin
    model = Nonparametric(;
        mean = x -> [1., 1.],
        amp_priors = fill(LogNormal(), 2),
        length_scale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
        noise_std_priors = fill(LogNormal(), 2),
    )
    data = ExperimentDataPrior([2.;2.;; 5.;5.;; 8.;8.;;], [2.;2.;; 5.;5.;; 8.;8.;;])

    @param_test BOSS.model_loglike begin
        @params deepcopy(model), deepcopy(data)
        @success (
            out isa Function,
            out((Float64[], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])) isa Real,
            out((Float64[], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])) < 0.,
            out((Float64[], [1.;1.;; 1.;1.;;], [1., 1.], [5., 5.])) > out((Float64[], [1.;1.;; 1.;1.;;], [1., 1.], [100., 100.])),
            out((Float64[], [1.;1.;; 1.;1.;;], [5., 5.], [1., 1.])) > out((Float64[], [1.;1.;; 1.;1.;;], [100., 100.], [1., 1.])),
            out((Float64[], [5.;5.;; 5.;5.;;], [1., 1.], [1., 1.])) > out((Float64[], [100.;100.;; 100.;100.;;], [1., 1.], [1., 1.])),
        )
    end
end

# TODO: Check the GP marginal data likelihood against the equiation in
# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7352306 section III. D.
@testset "data_loglike(model, X, Y, params)" begin
    model = Nonparametric(;
        mean = x -> [0.],
        amp_priors = fill(LogNormal(), 1),
        length_scale_priors = fill(product_distribution(fill(Dirac(1.), 1)), 1),
        noise_std_priors = fill(Dirac(0.1), 1),
    )

    @param_test BOSS.data_loglike begin
        @params deepcopy(model), [1.;; 2.;; 3.;;], [1.;; 2.;; 3.;;], (Float64[], [1.;;], [1.], [1.])
        @success out isa Real
    end

    t_length_scale(λ) = BOSS.data_loglike(deepcopy(model), [1.;; 2.;; 3.;;], [1.;; -1.;; 1.;;], (Float64[], λ, [1.], [0.]))
    @test t_length_scale([0.1;;]) > t_length_scale([1.;;]) > t_length_scale([10.;;])

    t_amplitude(α) = BOSS.data_loglike(deepcopy(model), [1.;; 2.;; 3.;;], [1.;; -1.;; 1.;;], (Float64[], [1.;;], α, [0.]))
    @test t_amplitude([1.]) > t_amplitude([0.1])
    @test t_amplitude([1.]) > t_amplitude([10.])

    t_noise_std(σ) = BOSS.data_loglike(deepcopy(model), [1.;; 2.;; 3.;;], [1.;; -1.;; 1.;;], (Float64[], [1.;;], [1.], σ))
    @test t_noise_std([1.]) > t_noise_std([0.1])
    @test t_noise_std([1.]) > t_noise_std([10.])

    t_data(X, Y) = BOSS.data_loglike(deepcopy(model), X, Y, (Float64[], [1.;;], [1.], [0.]))
    @test t_data([1.;; 2.;; 3;;], [99.9;; 100.;; 100.1;;]) > t_data([1.;; 2.;; 3;;], [99.;; 100.;; 101.;;]) > t_data([1.;; 2.;; 3;;], [90.;; 100.;; 110.;;])
end

@testset "model_params_loglike(model, params)" begin
    @param_test BOSS.model_params_loglike begin
        # TODO: Add different priors loaded from a collection.
        @params (
            Nonparametric(;
                length_scale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
                amp_priors = fill(LogNormal(), 2),
                noise_std_priors = fill(Dirac(0.1), 2),
            ),
            (Float64[], [1.;1.;; 1.;1.;;], [1., 2.], [0.1, 0.1]),
        )
        @success out isa Real

        @params (
            Nonparametric(;
                length_scale_priors = fill(product_distribution(fill(Dirac(1.), 2)), 2),
                amp_priors = fill(Dirac(1.), 2),
                noise_std_priors = fill(Dirac(0.1), 2),
            ),
            (Float64[], [1.;1.;; 1.;1.;;], [1., 1.], [0.1, 0.1]),
        )
        @success out == 0.

        @params (
            Nonparametric(;
                length_scale_priors = fill(product_distribution(fill(Dirac(1.), 2)), 2),
                amp_priors = fill(Dirac(1.), 2),
                noise_std_priors = fill(Dirac(0.1), 2),
            ),
            (Float64[], [1.;1.;; 5.;5.;;], [1., 1.], [0.1, 0.1]),
        )
        @params (
            Nonparametric(;
                length_scale_priors = fill(product_distribution(fill(Dirac(1.), 2)), 2),
                amp_priors = fill(Dirac(1.), 2),
                noise_std_priors = fill(Dirac(0.1), 2),
            ),
            (Float64[], [1.;1.;; 1.;1.;;], [1., 5.], [0.1, 0.1]),
        )
        @params (
            Nonparametric(;
                length_scale_priors = fill(product_distribution(fill(Dirac(1.), 2)), 2),
                amp_priors = fill(Dirac(1.), 2),
                noise_std_priors = fill(Dirac(0.1), 2),
            ),
            (Float64[], [1.;1.;; 1.;1.;;], [1., 1.], [0.1, 0.5]),
        )
        @success out == -Inf
    end
end
