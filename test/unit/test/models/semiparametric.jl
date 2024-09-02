
@testset "model_posterior(model, data)" begin
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = reduce(hcat, (x -> [sin(x[1]) + exp(x[2]), cos(x[1]) + exp(x[2])]).(eachcol(X)))
    
    problem = BOSS.BossProblem(;
        fitness = BOSS.LinFitness([1., 0.]),
        f = x -> x,
        domain = BOSS.Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        model = BOSS.Semiparametric(;
            parametric = BOSS.NonlinModel(;
                predict = (x, θ) -> [
                    θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
                    θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
                ],
                theta_priors = fill(BOSS.Normal(), 4),
            ),
            nonparametric = BOSS.Nonparametric(;
                amp_priors = fill(BOSS.LogNormal(), 2),
                length_scale_priors = fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2),
                noise_std_priors = fill(BOSS.Dirac(1e-4), 2),
            ),
        ),
        data = BOSS.ExperimentDataPrior(X, Y),
    )
    BOSS.estimate_parameters!(problem, BOSS.SamplingMAP(; samples=200, parallel=PARALLEL_TESTS); options=BOSS.BossOptions(; info=false))

    @param_test BOSS.model_posterior begin
        @params problem.model, problem.data
        @success (
            out isa Function,

            # vector
            out([2., 2.]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            length(out([2., 2.])[1]) == 2,
            length(out([2., 2.])[2]) == 2,
            all(out([2., 2.])[2] .< out([3., 3.])[2]),
            all(out([10., 10.])[2] .< out([11., 11.])[2]),

            # matrix
            out([1.;1.;; 2.;2.;; 3.;3.;;]) isa Tuple{<:AbstractMatrix{<:Real}, <:AbstractMatrix{<:Real}},
            size(out([1.;1.;; 2.;2.;; 3.;3.;;])[1]) == (2, 3),
            size(out([1.;1.;; 2.;2.;; 3.;3.;;])[2]) == (2, 3),
            isapprox.(out([1.;1.;; 2.;2.;; 3.;3.;;])[1][:,1], out([1., 1.])[1]; atol=1e-8) |> all,
            isapprox.(out([1.;1.;; 2.;2.;; 3.;3.;;])[1][:,2], out([2., 2.])[1]; atol=1e-8) |> all,
            isapprox.(out([1.;1.;; 2.;2.;; 3.;3.;;])[1][:,3], out([3., 3.])[1]; atol=1e-8) |> all,
            isapprox.(out([1.;1.;; 2.;2.;; 3.;3.;;])[2][:,1], out([1., 1.])[2]; atol=1e-8) |> all,
            isapprox.(out([1.;1.;; 2.;2.;; 3.;3.;;])[2][:,2], out([2., 2.])[2]; atol=1e-8) |> all,
            isapprox.(out([1.;1.;; 2.;2.;; 3.;3.;;])[2][:,3], out([3., 3.])[2]; atol=1e-8) |> all,
        )
    end
end

@testset "model_posterior_slice(model, data, slice)" begin
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = reduce(hcat, (x -> [sin(x[1]) + exp(x[2]), cos(x[1]) + exp(x[2])]).(eachcol(X)))
    
    problem = BOSS.BossProblem(;
        fitness = BOSS.LinFitness([1., 0.]),
        f = x -> x,
        domain = BOSS.Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        model = BOSS.Semiparametric(;
            parametric = BOSS.NonlinModel(;
                predict = (x, θ) -> [
                    θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
                    θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
                ],
                theta_priors = fill(BOSS.Normal(), 4),
            ),
            nonparametric = BOSS.Nonparametric(;
                amp_priors = fill(BOSS.LogNormal(), 2),
                length_scale_priors = fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2),
                noise_std_priors = fill(BOSS.Dirac(1e-4), 2),
            ),
        ),
        data = BOSS.ExperimentDataPrior(X, Y),
    )
    BOSS.estimate_parameters!(problem, BOSS.SamplingMAP(; samples=200, parallel=PARALLEL_TESTS); options=BOSS.BossOptions(; info=false))

    @param_test BOSS.model_posterior_slice begin
        @params problem.model, problem.data, 1
        @params problem.model, problem.data, 2
        @success (
            out isa Function,

            # vector
            out([2., 2.]) isa Tuple{<:Real, <:Real},
            out([2., 2.])[2] < out([3., 3.])[2],
            out([10., 10.])[2] < out([11., 11.])[2],

            # matrix
            out([1.;1.;; 2.;2.;; 3.;3.;;]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            length(out([1.;1.;; 2.;2.;; 3.;3.;;])[1]) == 3,
            length(out([1.;1.;; 2.;2.;; 3.;3.;;])[2]) == 3,
            isapprox(out([1.;1.;; 2.;2.;; 3.;3.;;])[1][1], out([1., 1.])[1]; atol=1e-8),
            isapprox(out([1.;1.;; 2.;2.;; 3.;3.;;])[1][2], out([2., 2.])[1]; atol=1e-8),
            isapprox(out([1.;1.;; 2.;2.;; 3.;3.;;])[1][3], out([3., 3.])[1]; atol=1e-8),
            isapprox(out([1.;1.;; 2.;2.;; 3.;3.;;])[2][1], out([1., 1.])[2]; atol=1e-8),
            isapprox(out([1.;1.;; 2.;2.;; 3.;3.;;])[2][2], out([2., 2.])[2]; atol=1e-8),
            isapprox(out([1.;1.;; 2.;2.;; 3.;3.;;])[2][3], out([3., 3.])[2]; atol=1e-8),
        )
    end
end

@testset "model_loglike(model, data)" begin
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = reduce(hcat, (x -> [sin(x[1]) + exp(x[2]), cos(x[1]) + exp(x[2])]).(eachcol(X)))
    
    model = BOSS.Semiparametric(;
        parametric = BOSS.NonlinModel(;
            predict = (x, θ) -> [
                θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
                θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
            ],
            theta_priors = fill(BOSS.Normal(), 4),
        ),
        nonparametric = BOSS.Nonparametric(;
            length_scale_priors = fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2),
            amp_priors = fill(BOSS.LogNormal(), 2),
            noise_std_priors = fill(BOSS.LogNormal(), 2),
        ),
    )
    data = BOSS.ExperimentDataPrior(X, Y)

    @param_test BOSS.model_loglike begin
        @params deepcopy(model), deepcopy(data)
        @success (
            out isa Function,
            out(([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])) isa Real,
            out(([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])) < 0.,
            out(([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [5., 5.])) > out(([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [100., 100.])),
            out(([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [5., 5.], [1., 1.])) > out(([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [100., 100.], [1., 1.])),
            out(([1., 1., 1., 1.], [5.;5.;; 5.;5.;;], [1., 1.], [1., 1.])) > out(([1., 1., 1., 1.], [100.;100.;; 100.;100.;;], [1., 1.], [1., 1.])),
            out(([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])) > out(([10., 10., 10., 10.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])),
            out(([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])) > out(([2., 2., 2., 2.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])),
            out(([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])) > out(([0.5, 0.5, 0.5, 0.5], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])),
        )
    end
end

@testset "data_loglike(model, X, Y, params)" begin
    parametric = BOSS.NonlinModel(;
        predict = (x, θ) -> [θ[1] * x[1]],
        theta_priors = fill(BOSS.Dirac(1.), 1),
    )
    nonparametric = BOSS.Nonparametric(;
        amp_priors = fill(BOSS.LogNormal(), 1),
        length_scale_priors = fill(BOSS.product_distribution(fill(BOSS.Dirac(1.), 1)), 1),
        noise_std_priors = fill(BOSS.Dirac(0.1), 1),
    )
    model = BOSS.Semiparametric(;
        parametric,
        nonparametric,
    )

    @param_test BOSS.data_loglike begin
        @params deepcopy(model), [1.;; 2.;; 3.;;], [1.;; 2.;; 3.;;], ([1.], [1.;;], [1.], [1.])
        @success out isa Real
    end

    t_theta(θ) = BOSS.data_loglike(deepcopy(model), [1.;; 2.;; 3.;;], [1.;; 2.;; 3.;;], (θ, [1.;;], [1.], [0.]))
    @test t_theta([1.]) > t_theta([0.1])
    @test t_theta([1.]) > t_theta([10.])

    t_length_scale(λ) = BOSS.data_loglike(deepcopy(model), [1.;; 2.;; 3.;;], [1.;; -1.;; 1.;;], ([0.], λ, [1.], [0.]))
    @test t_length_scale([0.1;;]) > t_length_scale([1.;;]) > t_length_scale([10.;;])

    t_amplitude(α) = BOSS.data_loglike(deepcopy(model), [1.;; 2.;; 3.;;], [1.;; -1.;; 1.;;], ([0.], [1.;;], α, [0.]))
    @test t_amplitude([1.]) > t_amplitude([0.1])
    @test t_amplitude([1.]) > t_amplitude([10.])

    t_noise_std(σ) = BOSS.data_loglike(deepcopy(model), [1.;; 2.;; 3.;;], [1.;; -1.;; 1.;;], ([0.], [1.;;], [1.], σ))
    @test t_noise_std([1.]) > t_noise_std([0.1])
    @test t_noise_std([1.]) > t_noise_std([10.])
end

@testset "model_params_loglike(model, params)" begin
    model = BOSS.Semiparametric(
        BOSS.NonlinModel(;
            predict = (x, θ) -> [
                θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
                θ[3] * cos(x[1]) + θ[4] * exp(x[2]),    
            ],
            theta_priors = fill(BOSS.Dirac(1.), 4),
        ),
        BOSS.Nonparametric(;
            length_scale_priors = fill(BOSS.product_distribution(fill(BOSS.Dirac(1.), 2)), 2),
            amp_priors = fill(BOSS.Dirac(1.), 2),
            noise_std_priors = fill(BOSS.Dirac(0.1), 2),
        ),
    )

    @param_test BOSS.model_params_loglike begin
        @params model, ([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [0.1, 0.1])
        @success out isa Real

        @params model, ([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [0.1, 0.1])
        @success out == 0.

        @params model, ([1., 5., 1., 5.], [1.;1.;; 1.;1.;;], [1., 1.], [0.1, 0.1])
        @params model, ([1., 1., 1., 1.], [1.;1.;; 5.;5.;;], [1., 1.], [0.1, 0.1])
        @params model, ([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 5.], [0.1, 0.1])
        @params model, ([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [0.1, 0.5])
        @success out == -Inf
    end
end
