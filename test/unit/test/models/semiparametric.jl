
@testset "model_posterior(model, params, data)" begin
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = reduce(hcat, (x -> [sin(x[1]) + exp(x[2]), cos(x[1]) + exp(x[2])]).(eachcol(X)))
    
    problem = BossProblem(;
        fitness = LinFitness([1., 0.]),
        f = x -> x,
        domain = Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        model = Semiparametric(;
            parametric = NonlinearModel(;
                predict = (x, θ) -> [
                    θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
                    θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
                ],
                theta_priors = fill(Normal(), 4),
            ),
            nonparametric = Nonparametric(;
                amplitude_priors = fill(LogNormal(), 2),
                lengthscale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
                noise_std_priors = fill(Dirac(1e-4), 2),
            ),
        ),
        data = ExperimentData(X, Y),
    )
    BOSS.estimate_parameters!(problem, SamplingMAP(; samples=200, parallel=PARALLEL_TESTS); options=BossOptions(; info=false))

    @param_test model_posterior begin
        @params problem.model, problem.params, problem.data
        @success (
            out isa Function,

            # vector
            out([2., 2.]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            length(out([2., 2.])[1]) == 2,
            length(out([2., 2.])[2]) == 2,
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

@testset "model_posterior_slice(model, params, data, slice)" begin
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = reduce(hcat, (x -> [sin(x[1]) + exp(x[2]), cos(x[1]) + exp(x[2])]).(eachcol(X)))
    
    problem = BossProblem(;
        fitness = LinFitness([1., 0.]),
        f = x -> x,
        domain = Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        model = Semiparametric(;
            parametric = NonlinearModel(;
                predict = (x, θ) -> [
                    θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
                    θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
                ],
                theta_priors = fill(Normal(), 4),
            ),
            nonparametric = Nonparametric(;
                amplitude_priors = fill(LogNormal(), 2),
                lengthscale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
                noise_std_priors = fill(Dirac(1e-4), 2),
            ),
        ),
        data = ExperimentData(X, Y),
    )
    BOSS.estimate_parameters!(problem, SamplingMAP(; samples=200, parallel=PARALLEL_TESTS); options=BossOptions(; info=false))

    @param_test model_posterior_slice begin
        @params problem.model, problem.params, problem.data, 1
        @params problem.model, problem.params, problem.data, 2
        @success (
            out isa Function,

            # vector
            out([2., 2.]) isa Tuple{<:Real, <:Real},
            out([2., 2.])[2] < out([3., 3.])[2],
            out([10., 10.])[2] < out([11., 11.])[2],

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

@testset "model_loglike(model, data)" begin
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = reduce(hcat, (x -> [sin(x[1]) + exp(x[2]), cos(x[1]) + exp(x[2])]).(eachcol(X)))
    
    model = Semiparametric(;
        parametric = NonlinearModel(;
            predict = (x, θ) -> [
                θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
                θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
            ],
            theta_priors = fill(Normal(), 4),
        ),
        nonparametric = Nonparametric(;
            lengthscale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
            amplitude_priors = fill(LogNormal(), 2),
            noise_std_priors = fill(LogNormal(), 2),
        ),
    )
    data = ExperimentData(X, Y)

    @param_test BOSS.model_loglike begin
        @params deepcopy(model), deepcopy(data)
        @success (
            out isa Function,
            out(SemiparametricParams([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])) isa Real,
            out(SemiparametricParams([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])) < 0.,
            out(SemiparametricParams([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [5., 5.])) > out(SemiparametricParams([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [100., 100.])),
            out(SemiparametricParams([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [5., 5.], [1., 1.])) > out(SemiparametricParams([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [100., 100.], [1., 1.])),
            out(SemiparametricParams([1., 1., 1., 1.], [5.;5.;; 5.;5.;;], [1., 1.], [1., 1.])) > out(SemiparametricParams([1., 1., 1., 1.], [100.;100.;; 100.;100.;;], [1., 1.], [1., 1.])),
            out(SemiparametricParams([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])) > out(SemiparametricParams([10., 10., 10., 10.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])),
            out(SemiparametricParams([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])) > out(SemiparametricParams([2., 2., 2., 2.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])),
            out(SemiparametricParams([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])) > out(SemiparametricParams([0.5, 0.5, 0.5, 0.5], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.])),
        )
    end
end

@testset "data_loglike(model, data)" begin
    parametric = NonlinearModel(;
        predict = (x, θ) -> [θ[1] * x[1]],
        theta_priors = fill(Dirac(1.), 1),
    )
    nonparametric = Nonparametric(;
        amplitude_priors = fill(LogNormal(), 1),
        lengthscale_priors = fill(product_distribution(fill(Dirac(1.), 1)), 1),
        noise_std_priors = fill(Dirac(0.1), 1),
    )
    model = Semiparametric(;
        parametric,
        nonparametric,
    )

    t_theta(θ) = SemiparametricParams(θ, [1.;;], [1.], [0.])
    t_length_scale(λ) = SemiparametricParams([0.], λ, [1.], [0.])
    t_amplitude(α) = SemiparametricParams([0.], [1.;;], α, [0.])
    t_noise_std(σ) = SemiparametricParams([0.], [1.;;], [1.], σ)

    @param_test BOSS.data_loglike begin
        @params deepcopy(model), ExperimentData([1.;; 2.;; 3.;;], [1.;; 2.;; 3.;;])
        @success out(SemiparametricParams([1.], [1.;;], [1.], [1.])) isa Real

        @params deepcopy(model), ExperimentData([1.;; 2.;; 3.;;], [1.;; 2.;; 3.;;])
        @success (
            out(t_theta([1.])) > out(t_theta([0.1])),
            out(t_theta([1.])) > out(t_theta([10.])),
        )

        @params deepcopy(model), ExperimentData([1.;; 2.;; 3.;;], [1.;; -1.;; 1.;;])
        @success out(t_length_scale([0.1;;])) > out(t_length_scale([1.;;])) > out(t_length_scale([10.;;]))

        @params deepcopy(model), ExperimentData([1.;; 2.;; 3.;;], [1.;; -1.;; 1.;;])
        @success (
            out(t_amplitude([1.])) > out(t_amplitude([0.1])),
            out(t_amplitude([1.])) > out(t_amplitude([10.])),
        )

        @params deepcopy(model), ExperimentData([1.;; 2.;; 3.;;], [1.;; -1.;; 1.;;])
        @success (
            out(t_noise_std([1.])) > out(t_noise_std([0.1])),
            out(t_noise_std([1.])) > out(t_noise_std([10.])),
        )
    end
end

@testset "params_loglike(model)" begin
    model = Semiparametric(
        NonlinearModel(;
            predict = (x, θ) -> [
                θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
                θ[3] * cos(x[1]) + θ[4] * exp(x[2]),    
            ],
            theta_priors = fill(Dirac(1.), 4),
        ),
        Nonparametric(;
            lengthscale_priors = fill(product_distribution(fill(Dirac(1.), 2)), 2),
            amplitude_priors = fill(Dirac(1.), 2),
            noise_std_priors = fill(Dirac(0.1), 2),
        ),
    )

    @param_test BOSS.params_loglike begin
        @params model
        @success (
            out(SemiparametricParams([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [0.1, 0.1])) isa Real,
            out(SemiparametricParams([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [0.1, 0.1])) == 0.,
            out(SemiparametricParams([1., 5., 1., 5.], [1.;1.;; 1.;1.;;], [1., 1.], [0.1, 0.1])) == -Inf,
            out(SemiparametricParams([1., 1., 1., 1.], [1.;1.;; 5.;5.;;], [1., 1.], [0.1, 0.1])) == -Inf,
            out(SemiparametricParams([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 5.], [0.1, 0.1])) == -Inf,
            out(SemiparametricParams([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [0.1, 0.5])) == -Inf,
        )
    end
end
