
@testset "(::LinModel)(x, θ)" begin
    @param_test BOSS.LinModel begin
        @params (
            (x) -> [[sin(x[1]), exp(x[1])]],
            [BOSS.Normal(), BOSS.Normal()],  # irrelevant
            nothing,
        )
        @success (
            isapprox(out([1.], [4., 5.]), [4. * sin(1.) + 5. * exp(1.)]; atol=1e-20),
            isapprox(out([4., 5.])([1.]), [4. * sin(1.) + 5. * exp(1.)]; atol=1e-20),
        )

        @params (
            (x) -> [[sin(x[1]), exp(x[1])]],
            [BOSS.Normal(), BOSS.Normal()],  # irrelevant
            [true],
        )
        @success (
            isapprox(out([4., 5.])([1.2]), [4. * sin(1.) + 5. * exp(1.)]; atol=1e-20),
            isapprox(out([4., 5.])([1.2]), [4. * sin(1.) + 5. * exp(1.)]; atol=1e-20),
        )
    end
end

@testset "(::NonlinModel)(x, θ)" begin
    @param_test BOSS.NonlinModel begin
        @params (
            (x, θ) -> [θ[1] * sin(θ[2] * x[1]) + exp(θ[3] * x[1])],
            fill(BOSS.Normal(), 3),  # irrelevant
            nothing,
        )
        @success (
            isapprox(out([1.], [3., 4., 5.]), [3. * sin(4. * 1.) + exp(5. * 1.)]; atol=1e-20),
            isapprox(out([3., 4., 5.])([1.]), [3. * sin(4. * 1.) + exp(5. * 1.)]; atol=1e-20),
        )

        @params (
            (x, θ) -> [θ[1] * sin(θ[2] * x[1]) + exp(θ[3] * x[1])],
            fill(BOSS.Normal(), 3),  # irrelevant
            [true],
        )
        @success (
            isapprox(out([1.2], [3., 4., 5.]), [3. * sin(4. * 1.) + exp(5. * 1.)]; atol=1e-20),
            isapprox(out([3., 4., 5.])([1.2]), [3. * sin(4. * 1.) + exp(5. * 1.)]; atol=1e-20),
        )
    end
end

@testset "Base.convert(::Type{NonlinModel}, ::LinModel)" begin
    @param_test Base.convert begin
        @params (
            BOSS.NonlinModel,
            BOSS.LinModel(;
                lift = (x) -> [[sin(x[1]), exp(x[1])]],
                param_priors = [BOSS.Normal(), BOSS.Normal()],
                discrete = nothing,
            ),
        )
        @success (
            out isa BOSS.NonlinModel,
            out([1.], [4., 5.]) == in[2]([1.], [4., 5.]),
            out.param_priors == in[2].param_priors,
        )

        @params (
            BOSS.NonlinModel,
            BOSS.LinModel(;
                lift = (x) -> [[sin(x[1]), exp(x[1])]],
                param_priors = [BOSS.Normal(), BOSS.Normal()],
                discrete = [true],
            ),
        )
        @success (
            out isa BOSS.NonlinModel,
            out([1.2], [4., 5.]) == in[2]([1.2], [4., 5.]),
            out.param_priors == in[2].param_priors,
        )
    end
end

@testset "make_discrete(model, discrete)" begin
    @param_test BOSS.make_discrete begin
        @params (
            BOSS.LinModel(;
                lift = (x) -> [[sin(x[1]), exp(x[2])]],
                param_priors = [BOSS.Normal(), BOSS.Normal()],
                discrete = nothing,
            ),
            [false, true],
        )
        @params (
            BOSS.NonlinModel(;
                predict = (x, θ) -> [θ[1] * sin(x[1]) + θ[2] * exp(x[2])],
                param_priors = [BOSS.Normal(), BOSS.Normal()],
                discrete = nothing,
            ),
            [false, true],
        )
        @success (
            out([1., 1.], [4., 5.]) == in[1]([1., 1.], [4., 5.]),
            out([1.2, 1.2], [4., 5.]) == in[1]([1.2, 1.], [4., 5.]),
            out([1.2, 1.2], [4., 5.]) != in[1]([1.2, 1.2], [4., 5.]),
        )

        @params (
            BOSS.LinModel(;
                lift = (x) -> [[sin(x[1]), exp(x[2])]],
                param_priors = [BOSS.Normal(), BOSS.Normal()],
                discrete = [false, true],
            ),
            [false, true],
        )
        @params (
            BOSS.NonlinModel(;
                predict = (x, θ) -> [θ[1] * sin(x[1]) + θ[2] * exp(x[2])],
                param_priors = [BOSS.Normal(), BOSS.Normal()],
                discrete = [false, true],
            ),
            [false, true],
        )
        @success (
            out([1., 1.], [4., 5.]) == in[1]([1., 1.], [4., 5.]),
            out([1.2, 1.2], [4., 5.]) == in[1]([1.2, 1.2], [4., 5.]),
        )

        @params (
            BOSS.LinModel(;
                lift = (x) -> [[sin(x[1]), exp(x[2])]],
                param_priors = [BOSS.Normal(), BOSS.Normal()],
                discrete = nothing,
            ),
            [false, false],
        )
        @params (
            BOSS.NonlinModel(;
                predict = (x, θ) -> [θ[1] * sin(x[1]) + θ[2] * exp(x[2])],
                param_priors = [BOSS.Normal(), BOSS.Normal()],
                discrete = nothing,
            ),
            [false, false],
        )
        @success (
            out([1.2, 1.2], [4., 5.]) != in[1]([1., 1.], [4., 5.]),
            out([1.2, 1.2], [4., 5.]) == in[1]([1.2, 1.2], [4., 5.]),
        )
    end
end

@testset "model_posterior(model, data)" begin
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = reduce(hcat, (x -> [sin(x[1]) + exp(x[2]), cos(x[1]) + exp(x[2])]).(eachcol(X)))

    problem(model) = BOSS.OptimizationProblem(;
        fitness = BOSS.LinFitness([1., 0.]),
        f = x -> x,
        domain = BOSS.Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        model,
        noise_var_priors = fill(BOSS.Dirac(1e-8), 2),
        data = BOSS.ExperimentDataPrior(X, Y),
    )

    lin_model = BOSS.LinModel(;
        lift = (x) -> [
            [sin(x[1]), exp(x[2])],
            [cos(x[1]), exp(x[2])],
        ],
        param_priors = fill(BOSS.Normal(), 4),
        discrete = nothing,
    )
    nonlin_model = BOSS.NonlinModel(;
        predict = (x, θ) -> [
            θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
            θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
        ],
        param_priors = fill(BOSS.Normal(), 4),
        discrete = nothing,
    )

    problem_lin = problem(lin_model)
    problem_nonlin = problem(nonlin_model)
    BOSS.estimate_parameters!(problem_lin, BOSS.SamplingMLE(; samples=200, parallel=PARALLEL_TESTS); options=BOSS.BossOptions(; info=false))
    BOSS.estimate_parameters!(problem_nonlin, BOSS.SamplingMLE(; samples=200, parallel=PARALLEL_TESTS); options=BOSS.BossOptions(; info=false))

    @param_test BOSS.model_posterior begin
        @params problem_lin.model, problem_lin.data
        @params problem_nonlin.model, problem_nonlin.data
        @success (
            out isa Function,
            out([2., 2.]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            length(out([2., 2.])[1]) == 2,
            length(out([2., 2.])[2]) == 2,
            out([2., 2.])[2] == [1e-8, 1e-8],
            all(out([2., 2.])[2] == out([3., 3.])[2]),
            all(out([10., 10.])[2] == out([11., 11.])[2]),
        )
    end
end

@testset "model_loglike(model, noise_var_priors, data)" begin
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = reduce(hcat, (x -> [sin(x[1]) + exp(x[2]), cos(x[1]) + exp(x[2])]).(eachcol(X)))

    lin_model = BOSS.LinModel(;
        lift = (x) -> [
            [sin(x[1]), exp(x[2])],
            [cos(x[1]), exp(x[2])],
        ],
        param_priors = fill(BOSS.Normal(), 4),
        discrete = nothing,
    )
    nonlin_model = BOSS.NonlinModel(;
        predict = (x, θ) -> [
            θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
            θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
        ],
        param_priors = fill(BOSS.Normal(), 4),
        discrete = nothing,
    )
    noise_var_priors = fill(BOSS.LogNormal(1., 1.), 2)
    data = BOSS.ExperimentDataPrior(X, Y)

    @param_test BOSS.model_loglike begin
        @params lin_model, deepcopy(noise_var_priors), deepcopy(data)
        @params nonlin_model, deepcopy(noise_var_priors), deepcopy(data)
        @success (
            out isa Function,
            out([1., 1., 1., 1.], Float64[;;], [1., 1.]) isa Real,
            out([1., 1., 1., 1.], Float64[;;], [1., 1.]) < 0.,
            out([1., 1., 1., 1.], Float64[;;], [1., 1.]) > out([10., 10., 10., 10.], Float64[;;], [1., 1.]),
            out([1., 1., 1., 1.], Float64[;;], [5., 5.]) > out([1., 1., 1., 1.], Float64[;;], [100., 100.]),
            out([1., 1., 1., 1.], Float64[;;], [1., 1.]) > out([2., 2., 2., 2.], Float64[;;], [1., 1.]),
            out([1., 1., 1., 1.], Float64[;;], [1., 1.]) > out([0.5, 0.5, 0.5, 0.5], Float64[;;], [1., 1.]),
        )
    end
end

@testset "model_params_loglike(model, θ)" begin
    @param_test BOSS.model_params_loglike begin
        # TODO: Add different priors loaded from a collection.
        @params (
            BOSS.LinModel(;
                lift = (x) -> [[sin(x[1]), exp(x[2])]],
                param_priors = [BOSS.Normal(), BOSS.Normal()],
            ),
            [1., 1.],
        )
        @params (
            BOSS.NonlinModel(;
                predict = (x, θ) -> [θ[1] * sin(x[1]) + θ[2] * exp(x[2])],
                param_priors = [BOSS.Normal(), BOSS.Normal()],
            ),
            [1., 1.],
        )
        @success isapprox(
            out,
            sum((BOSS.logpdf(in[1].param_priors[i], in[2][i]) for i in 1:2));
            atol=1e-20,
        )

        @params (
            BOSS.LinModel(;
                lift = (x) -> [[sin(x[1]), exp(x[2])]],
                param_priors = fill(BOSS.Dirac(1.), 2),
            ),
            [1., 1.],
        )
        @params (
            BOSS.NonlinModel(;
                predict = (x, θ) -> [θ[1] * sin(x[1]) + θ[2] * exp(x[2])],
                param_priors = fill(BOSS.Dirac(1.), 2),
            ),
            [1., 1.],
        )
        @success out == 0.

        @params (
            BOSS.LinModel(;
                lift = (x) -> [[sin(x[1]), exp(x[2])]],
                param_priors = fill(BOSS.Dirac(1.), 2),
            ),
            [1., 5.],
        )
        @params (
            BOSS.NonlinModel(;
                predict = (x, θ) -> [θ[1] * sin(x[1]) + θ[2] * exp(x[2])],
                param_priors = fill(BOSS.Dirac(1.), 2),
            ),
            [1., 5.],
        )
        @success out == -Inf
    end
end

@testset "model_data_loglike(model, θ, noise_vars, X, Y)" begin
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = reduce(hcat, (x -> [sin(x[1]) + exp(x[2]), cos(x[1]) + exp(x[2])]).(eachcol(X)))

    lin_model = BOSS.LinModel(;
        lift = (x) -> [
            [sin(x[1]), exp(x[2])],
            [cos(x[1]), exp(x[2])],
        ],
        param_priors = fill(BOSS.Dirac(1.), 4),
    )
    nonlin_model = BOSS.NonlinModel(;
        predict = (x, θ) -> [
            θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
            θ[3] * cos(x[1]) + θ[4] * exp(x[2]),    
        ],
        param_priors = fill(BOSS.Dirac(1.), 4),
    )

    @param_test BOSS.model_data_loglike begin
        @params deepcopy(lin_model), [1., 1., 1., 1.], [1., 1.], deepcopy(X), deepcopy(Y)
        @params deepcopy(nonlin_model), [1., 1., 1., 1.], [1., 1.], deepcopy(X), deepcopy(Y)
        @success out < 0.
    end

    t_θ(θ; model) = BOSS.model_data_loglike(deepcopy(model), θ, [1., 1.], deepcopy(X), deepcopy(Y))
    @param_test model -> (θ -> t_θ(θ; model)) begin
        @params lin_model
        @params nonlin_model
        @success (
            out([1., 1., 1., 1.]) > out([2., 2., 2., 2.]),
            out([1., 1., 1., 1.]) > out([0.5, 0.5, 0.5, 0.5]),
        )
    end

    t_noise_vars(σ2; model, Y) = BOSS.model_data_loglike(deepcopy(model), [1., 1., 1., 1.], σ2, deepcopy(X), deepcopy(Y))
    @param_test (model, Y) -> (σ2 -> t_noise_vars(σ2; model, Y)) begin
        @params lin_model, Y
        @params nonlin_model, Y
        @success out([0.1, 0.1]) > out([1., 1.]) > out([10., 10.])  # because model fits the data

        @params lin_model, [1.;1.;; 2.;2.;; 3.;3.;;]
        @params nonlin_model, [1.;1.;; 2.;2.;; 3.;3.;;]
        @success out([0.1, 0.1]) < out([1., 1.]) < out([10., 10.])  # because model does not fit the data
    end

    t_data(X, Y; model) = BOSS.model_data_loglike(deepcopy(model), [1., 1., 1., 1.], [1., 1.], deepcopy(X), deepcopy(Y))
    @param_test model -> ((X, Y) -> t_data(X, Y; model)) begin
       @params lin_model
       @params nonlin_model
       @success out(X, Y) > out(X, [1.;1.;; 2.;2.;; 3.;3.;;])
    end
end
