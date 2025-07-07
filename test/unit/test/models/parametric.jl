
@testset "(::LinearModel)(x, θ)" begin
    @param_test LinearModel begin
        @params (
            (x) -> [[sin(x[1]), exp(x[1])]],
            [Normal(), Normal()],  # irrelevant
            nothing,
            nothing,
        )
        @success out()([1.], [4., 5.]) == out([4., 5.])([1.]) == out([1.], [4., 5.]) ≈ [4. * sin(1.) + 5. * exp(1.)]

        @params (
            (x) -> [[sin(x[1]), exp(x[1])]],
            [Normal(), Normal()],  # irrelevant
            [true],
            nothing,
        )
        @success (
            out([1.2], [4., 5.]) == out([1.], [4., 5.]),
            out()([1.2], [4., 5.]) == out([4., 5.])([1.2]) == out([1.2], [4., 5.]) ≈ [4. * sin(1.) + 5. * exp(1.)],
        )
    end
end

@testset "(::NonlinearModel)(x, θ)" begin
    @param_test NonlinearModel begin
        @params (
            (x, θ) -> [θ[1] * sin(θ[2] * x[1]) + exp(θ[3] * x[1])],
            fill(Normal(), 3),  # irrelevant
            nothing,
            nothing,
        )
        @success out()([1.], [3., 4., 5.]) == out([3., 4., 5.])([1.]) == out([1.], [3., 4., 5.]) ≈ [3. * sin(4. * 1.) + exp(5. * 1.)]

        @params (
            (x, θ) -> [θ[1] * sin(θ[2] * x[1]) + exp(θ[3] * x[1])],
            fill(Normal(), 3),  # irrelevant
            [true],
            nothing,
        )
        @success (
            out([1.2], [3., 4., 5.]) == out([1.], [3., 4., 5.]),
            out()([1.2], [3., 4., 5.]) == out([3., 4., 5.])([1.2]) == out([1.2], [3., 4., 5.]) ≈ [3. * sin(4. * 1.) + exp(5. * 1.)],
        )
    end
end

@testset "Base.convert(::Type{NonlinearModel}, ::LinearModel)" begin
    @param_test Base.convert begin
        @params (
            NonlinearModel,
            LinearModel(;
                lift = (x) -> [[sin(x[1]), exp(x[1])]],
                theta_priors = [Normal(), Normal()],
                noise_std_priors = [Dirac(0.1)],
            ),
        )
        @success (
            out isa NonlinearModel,
            out([1.], [4., 5.]) == in[2]([1.], [4., 5.]),
            out.theta_priors == in[2].theta_priors,
            out.noise_std_priors == in[2].noise_std_priors,
        )

        @params (
            NonlinearModel,
            LinearModel(;
                lift = (x) -> [[sin(x[1]), exp(x[1])]],
                theta_priors = [Normal(), Normal()],
                discrete = [true],
                noise_std_priors = [Dirac(0.1)],
            ),
        )
        @success (
            out isa NonlinearModel,
            out([1.2], [4., 5.]) == in[2]([1.2], [4., 5.]),
            out.theta_priors == in[2].theta_priors,
            out.noise_std_priors == in[2].noise_std_priors,
        )
    end
end

@testset "make_discrete(model, discrete)" begin
    @param_test BOSS.make_discrete begin
        @params (
            LinearModel(;
                lift = (x) -> [[sin(x[1]), exp(x[2])]],
                theta_priors = [Normal(), Normal()],
            ),
            [false, true],
        )
        @params (
            NonlinearModel(;
                predict = (x, θ) -> [θ[1] * sin(x[1]) + θ[2] * exp(x[2])],
                theta_priors = [Normal(), Normal()],
            ),
            [false, true],
        )
        @success (
            out.discrete == in[2],
            out([1., 1.], [4., 5.]) == in[1]([1., 1.], [4., 5.]),
            out([1.2, 1.2], [4., 5.]) == in[1]([1.2, 1.], [4., 5.]),
            out([1.2, 1.2], [4., 5.]) != in[1]([1.2, 1.2], [4., 5.]),
        )

        @params (
            LinearModel(;
                lift = (x) -> [[sin(x[1]), exp(x[2])]],
                theta_priors = [Normal(), Normal()],
                discrete = [false, true],
            ),
            [false, true],
        )
        @params (
            NonlinearModel(;
                predict = (x, θ) -> [θ[1] * sin(x[1]) + θ[2] * exp(x[2])],
                theta_priors = [Normal(), Normal()],
                discrete = [false, true],
            ),
            [false, true],
        )
        @success (
            out([1., 1.], [4., 5.]) == in[1]([1., 1.], [4., 5.]),
            out([1.2, 1.2], [4., 5.]) == in[1]([1.2, 1.2], [4., 5.]),
        )

        @params (
            LinearModel(;
                lift = (x) -> [[sin(x[1]), exp(x[2])]],
                theta_priors = [Normal(), Normal()],
            ),
            [false, false],
        )
        @params (
            NonlinearModel(;
                predict = (x, θ) -> [θ[1] * sin(x[1]) + θ[2] * exp(x[2])],
                theta_priors = [Normal(), Normal()],
            ),
            [false, false],
        )
        @success (
            out.discrete == in[2],
            out([1.2, 1.2], [4., 5.]) != in[1]([1., 1.], [4., 5.]),
            out([1.2, 1.2], [4., 5.]) == in[1]([1.2, 1.2], [4., 5.]),
        )
    end
end

@testset "model_posterior(model, params, data)" begin
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = reduce(hcat, (x -> [sin(x[1]) + exp(x[2]), cos(x[1]) + exp(x[2])]).(eachcol(X)))

    problem(model) = BossProblem(;
        f = x -> x,
        domain = Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        acquisition = ExpectedImprovement(;
            fitness = LinFitness([1., 0.]),
        ),
        model,
        data = ExperimentData(X, Y),
    )

    lin_model = LinearModel(;
        lift = (x) -> [
            [sin(x[1]), exp(x[2])],
            [cos(x[1]), exp(x[2])],
        ],
        theta_priors = fill(Normal(), 4),
        noise_std_priors = fill(Dirac(1e-4), 2),
    )
    nonlin_model = NonlinearModel(;
        predict = (x, θ) -> [
            θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
            θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
        ],
        theta_priors = fill(Normal(), 4),
        noise_std_priors = fill(Dirac(1e-4), 2),
    )

    problem_lin = problem(lin_model)
    problem_nonlin = problem(nonlin_model)
    BOSS.estimate_parameters!(problem_lin, SamplingMAP(; samples=200, parallel=PARALLEL_TESTS); options=BossOptions(; info=false))
    BOSS.estimate_parameters!(problem_nonlin, SamplingMAP(; samples=200, parallel=PARALLEL_TESTS); options=BossOptions(; info=false))

    @param_test model_posterior begin
        @params problem_lin.model, problem_lin.params, problem_lin.data
        @params problem_nonlin.model, problem_nonlin.params, problem_nonlin.data
        @success (
            # vector
            mean(out, [2., 2.]) isa AbstractVector{<:Real},
            std(out, [2., 2.]) isa AbstractVector{<:Real},
            var(out, [2., 2.]) isa AbstractVector{<:Real},
            mean_and_std(out, [2., 2.]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            mean_and_var(out, [2., 2.]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            size(mean(out, [2., 2.])) == (2,),
            size(std(out, [2., 2.])) == (2,),
            size(var(out, [2., 2.])) == (2,),

            isapprox(mean(out, [2., 2.]), mean_and_std(out, [2., 2.])[1]; atol=1e-8),
            isapprox(mean(out, [2., 2.]), mean_and_var(out, [2., 2.])[1]; atol=1e-8),
            isapprox(std(out, [2., 2.]), mean_and_std(out, [2., 2.])[2]; atol=1e-8),
            isapprox(var(out, [2., 2.]), mean_and_var(out, [2., 2.])[2]; atol=1e-8),
            std(out, [2., 2.]) == [1e-4, 1e-4],
            std(out, [2., 2.]) == std(out, [3., 3.]),
            std(out, [10., 10.]) == std(out, [11., 11.]),

            # matrix
            mean(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractMatrix{<:Real},
            std(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractMatrix{<:Real},
            var(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractMatrix{<:Real},
            mean_and_std(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa Tuple{<:AbstractMatrix{<:Real}, <:AbstractMatrix{<:Real}},
            mean_and_var(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa Tuple{<:AbstractMatrix{<:Real}, <:AbstractMatrix{<:Real}},
            size(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (3, 2),
            size(std(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (3, 2),
            size(var(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (3, 2),
            
            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_std(out, [1.;1.;; 2.;2.;; 3.;3.;;])[1]; atol=1e-8),
            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_var(out, [1.;1.;; 2.;2.;; 3.;3.;;])[1]; atol=1e-8),
            isapprox(std(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_std(out, [1.;1.;; 2.;2.;; 3.;3.;;])[2]; atol=1e-8),
            isapprox(var(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_var(out, [1.;1.;; 2.;2.;; 3.;3.;;])[2]; atol=1e-8),
            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;])[1,:], mean(out, [1., 1.]); atol=1e-8),
            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;])[2,:], mean(out, [2., 2.]); atol=1e-8),
            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;])[3,:], mean(out, [3., 3.]); atol=1e-8),
            isapprox(var(out, [1.;1.;; 2.;2.;; 3.;3.;;])[1,:], var(out, [1., 1.]); atol=1e-8),
            isapprox(var(out, [1.;1.;; 2.;2.;; 3.;3.;;])[2,:], var(out, [2., 2.]); atol=1e-8),
            isapprox(var(out, [1.;1.;; 2.;2.;; 3.;3.;;])[3,:], var(out, [3., 3.]); atol=1e-8),

            # single-element matrix
            mean(out, [1.;1.;;]) isa AbstractMatrix{<:Real},
            std(out, [1.;1.;;]) isa AbstractMatrix{<:Real},
            var(out, [1.;1.;;]) isa AbstractMatrix{<:Real},
            mean_and_std(out, [1.;1.;;]) isa Tuple{<:AbstractMatrix{<:Real}, <:AbstractMatrix{<:Real}},
            mean_and_var(out, [1.;1.;;]) isa Tuple{<:AbstractMatrix{<:Real}, <:AbstractMatrix{<:Real}},
            size(mean(out, [1.;1.;;])) == (1, 2),
            size(std(out, [1.;1.;;])) == (1, 2),
            size(var(out, [1.;1.;;])) == (1, 2),

            isapprox(mean(out, [1.;1.;;]), mean_and_std(out, [1.;1.;;])[1]; atol=1e-8),
            isapprox(mean(out, [1.;1.;;]), mean_and_var(out, [1.;1.;;])[1]; atol=1e-8),
            isapprox(std(out, [1.;1.;;]), mean_and_std(out, [1.;1.;;])[2]; atol=1e-8),
            isapprox(var(out, [1.;1.;;]), mean_and_var(out, [1.;1.;;])[2]; atol=1e-8),
        )
    end

    # Test that covariance is not defined
    post = model_posterior(lin_model, problem_lin.params, problem_lin.data)
    @test_throws MethodError cov(post, [2., 2.])
    @test_throws MethodError cov(post, [1.;1.;; 2.;2.;; 3.;3.;;])
    @test_throws MethodError mean_and_cov(post, [2., 2.])
    @test_throws MethodError mean_and_cov(post, [1.;1.;; 2.;2.;; 3.;3.;;])
end

@testset "model_posterior_slice(model, params, data, slice)" begin
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = reduce(hcat, (x -> [sin(x[1]) + exp(x[2]), cos(x[1]) + exp(x[2])]).(eachcol(X)))

    problem(model) = BossProblem(;
        f = x -> x,
        domain = Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        acquisition = ExpectedImprovement(;
            fitness = LinFitness([1., 0.]),
        ),
        model,
        data = ExperimentData(X, Y),
    )

    lin_model = LinearModel(;
        lift = (x) -> [
            [sin(x[1]), exp(x[2])],
            [cos(x[1]), exp(x[2])],
        ],
        theta_priors = fill(Normal(), 4),
        noise_std_priors = fill(Dirac(1e-4), 2),
    )
    nonlin_model = NonlinearModel(;
        predict = (x, θ) -> [
            θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
            θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
        ],
        theta_priors = fill(Normal(), 4),
        noise_std_priors = fill(Dirac(1e-4), 2),
    )

    problem_lin = problem(lin_model)
    problem_nonlin = problem(nonlin_model)
    BOSS.estimate_parameters!(problem_lin, SamplingMAP(; samples=200, parallel=PARALLEL_TESTS); options=BossOptions(; info=false))
    BOSS.estimate_parameters!(problem_nonlin, SamplingMAP(; samples=200, parallel=PARALLEL_TESTS); options=BossOptions(; info=false))

    @param_test model_posterior_slice begin
        @params problem_lin.model, problem_lin.params, problem_lin.data, 1
        @params problem_lin.model, problem_lin.params, problem_lin.data, 2
        @params problem_nonlin.model, problem_nonlin.params, problem_nonlin.data, 1
        @params problem_nonlin.model, problem_nonlin.params, problem_nonlin.data, 2
        @success (            
            # vector
            mean(out, [2., 2.]) isa Real,
            std(out, [2., 2.]) isa Real,
            var(out, [2., 2.]) isa Real,
            mean_and_std(out, [2., 2.]) isa Tuple{<:Real, <:Real},
            mean_and_var(out, [2., 2.]) isa Tuple{<:Real, <:Real},

            isapprox(mean(out, [2., 2.]), mean_and_std(out, [2., 2.])[1]; atol=1e-8),
            isapprox(mean(out, [2., 2.]), mean_and_var(out, [2., 2.])[1]; atol=1e-8),
            isapprox(std(out, [2., 2.]), mean_and_std(out, [2., 2.])[2]; atol=1e-8),
            isapprox(var(out, [2., 2.]), mean_and_var(out, [2., 2.])[2]; atol=1e-8),
            std(out, [2., 2.]) == 1e-4,
            std(out, [2., 2.]) == std(out, [3., 3.]),
            std(out, [10., 10.]) == std(out, [11., 11.]),

            # matrix
            mean(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractVector{<:Real},
            std(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractVector{<:Real},
            var(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractVector{<:Real},
            mean_and_std(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            mean_and_var(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            size(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (3,),
            size(std(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (3,),
            size(var(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (3,),

            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;])[1], mean(out, [1., 1.]); atol=1e-8),
            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;])[2], mean(out, [2., 2.]); atol=1e-8),
            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;])[3], mean(out, [3., 3.]); atol=1e-8),
            isapprox(var(out, [1.;1.;; 2.;2.;; 3.;3.;;])[1], var(out, [1., 1.]); atol=1e-8),
            isapprox(var(out, [1.;1.;; 2.;2.;; 3.;3.;;])[2], var(out, [2., 2.]); atol=1e-8),
            isapprox(var(out, [1.;1.;; 2.;2.;; 3.;3.;;])[3], var(out, [3., 3.]); atol=1e-8),

            # single-element matrix
            mean(out, [1.;1.;;]) isa AbstractVector{<:Real},
            std(out, [1.;1.;;]) isa AbstractVector{<:Real},
            var(out, [1.;1.;;]) isa AbstractVector{<:Real},
            mean_and_std(out, [1.;1.;;]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            mean_and_var(out, [1.;1.;;]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            size(mean(out, [1.;1.;;])) == (1,),
            size(std(out, [1.;1.;;])) == (1,),
            size(var(out, [1.;1.;;])) == (1,),

            isapprox(mean(out, [1.;1.;;]), mean_and_std(out, [1.;1.;;])[1]; atol=1e-8),
            isapprox(mean(out, [1.;1.;;]), mean_and_var(out, [1.;1.;;])[1]; atol=1e-8),
            isapprox(std(out, [1.;1.;;]), mean_and_std(out, [1.;1.;;])[2]; atol=1e-8),
            isapprox(var(out, [1.;1.;;]), mean_and_var(out, [1.;1.;;])[2]; atol=1e-8),
        )
    end

    # Test that covariance is not defined
    post = model_posterior_slice(lin_model, problem_lin.params, problem_lin.data, 1)
    @test_throws MethodError cov(post, [2., 2.])
    @test_throws MethodError cov(post, [1.;1.;; 2.;2.;; 3.;3.;;])
    @test_throws MethodError mean_and_cov(post, [2., 2.])
    @test_throws MethodError mean_and_cov(post, [1.;1.;; 2.;2.;; 3.;3.;;])
end

@testset "model_loglike(model, data)" begin
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = reduce(hcat, (x -> [sin(x[1]) + exp(x[2]), cos(x[1]) + exp(x[2])]).(eachcol(X)))

    lin_model = LinearModel(;
        lift = (x) -> [
            [sin(x[1]), exp(x[2])],
            [cos(x[1]), exp(x[2])],
        ],
        theta_priors = fill(Normal(), 4),
        noise_std_priors = fill(LogNormal(), 2),
    )
    nonlin_model = NonlinearModel(;
        predict = (x, θ) -> [
            θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
            θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
        ],
        theta_priors = fill(Normal(), 4),
        noise_std_priors = fill(LogNormal(), 2),
    )
    data = ExperimentData(X, Y)

    @param_test BOSS.model_loglike begin
        @params lin_model, deepcopy(data)
        @params nonlin_model, deepcopy(data)
        @success (
            out isa Function,
            out(ParametricParams([1., 1., 1., 1.], [1., 1.])) isa Real,
            out(ParametricParams([1., 1., 1., 1.], [1., 1.])) < 0.,
            out(ParametricParams([1., 1., 1., 1.], [1., 1.])) > out(ParametricParams([10., 10., 10., 10.], [1., 1.])),
            out(ParametricParams([1., 1., 1., 1.], [5., 5.])) > out(ParametricParams([1., 1., 1., 1.], [100., 100.])),
            out(ParametricParams([1., 1., 1., 1.], [1., 1.])) > out(ParametricParams([2., 2., 2., 2.], [1., 1.])),
            out(ParametricParams([1., 1., 1., 1.], [1., 1.])) > out(ParametricParams([0.5, 0.5, 0.5, 0.5], [1., 1.])),
        )
    end
end

@testset "data_loglike(model, data)" begin
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = reduce(hcat, (x -> [sin(x[1]) + exp(x[2]), cos(x[1]) + exp(x[2])]).(eachcol(X)))
    data = ExperimentData(X, Y)
    bad_data = ExperimentData(X, [1.;1.;; 2.;2.;; 3.;3.;;])

    lin_model = LinearModel(;
        lift = (x) -> [
            [sin(x[1]), exp(x[2])],
            [cos(x[1]), exp(x[2])],
        ],
        theta_priors = fill(Dirac(1.), 4),
        noise_std_priors = fill(Dirac(0.1), 2),
    )
    nonlin_model = NonlinearModel(;
        predict = (x, θ) -> [
            θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
            θ[3] * cos(x[1]) + θ[4] * exp(x[2]),    
        ],
        theta_priors = fill(Dirac(1.), 4),
        noise_std_priors = fill(Dirac(0.1), 2),
    )

    t_theta(θ) = ParametricParams(θ, [1., 1.])
    t_noise_std(σ) = ParametricParams([1., 1., 1., 1.], σ)

    @param_test BOSS.data_loglike begin
        @params deepcopy(lin_model), deepcopy(data)
        @params deepcopy(nonlin_model), deepcopy(data)
        @success out(ParametricParams([1., 1., 1., 1.], [1., 1.])) < 0.

        @params deepcopy(lin_model), deepcopy(data)
        @params deepcopy(nonlin_model), deepcopy(data)
        @success (
            out(t_theta([1., 1., 1., 1.])) > out(t_theta([2., 2., 2., 2.])),
            out(t_theta([1., 1., 1., 1.])) > out(t_theta([0.5, 0.5, 0.5, 0.5])),
        )

        @params deepcopy(lin_model), deepcopy(data)
        @params deepcopy(nonlin_model), deepcopy(data)
        @success out(t_noise_std([0.1, 0.1])) > out(t_noise_std([1., 1.])) > out(t_noise_std([10., 10.])) # because model fits the data

        @params deepcopy(lin_model), deepcopy(bad_data)
        @params deepcopy(nonlin_model), deepcopy(bad_data)
        @success out(t_noise_std([0.1, 0.1])) < out(t_noise_std([1., 1.])) < out(t_noise_std([10., 10.])) # because model does not fit the data
    end

    t_data(X, Y; model) = BOSS.data_loglike(deepcopy(model), ExperimentData(deepcopy(X), deepcopy(Y)))(ParametricParams([1., 1., 1., 1.], [1., 1.]))
    @param_test model -> ((X, Y) -> t_data(X, Y; model)) begin
       @params lin_model
       @params nonlin_model
       @success out(X, Y) > out(X, [1.;1.;; 2.;2.;; 3.;3.;;])
    end
end

@testset "params_loglike(model)" begin
    @param_test BOSS.params_loglike begin
        # TODO: Add different priors loaded from a collection.
        @params LinearModel(;
                lift = (x) -> [[sin(x[1]), exp(x[2])]],
                theta_priors = [Normal(), Normal()],
                noise_std_priors = fill(Dirac(0.1), 2),
            )
        @params NonlinearModel(;
                predict = (x, θ) -> [θ[1] * sin(x[1]) + θ[2] * exp(x[2])],
                theta_priors = [Normal(), Normal()],
                noise_std_priors = fill(Dirac(0.1), 2),
            )
        @success out(ParametricParams([1., 1.], [0.1, 0.1])) isa Real

        @params LinearModel(;
                lift = (x) -> [[sin(x[1]), exp(x[2])]],
                theta_priors = fill(Dirac(1.), 2),
                noise_std_priors = fill(Dirac(0.1), 2),
            )
        @params NonlinearModel(;
                predict = (x, θ) -> [θ[1] * sin(x[1]) + θ[2] * exp(x[2])],
                theta_priors = fill(Dirac(1.), 2),
                noise_std_priors = fill(Dirac(0.1), 2),
            )
        @success out(ParametricParams([1., 1.], [0.1, 0.1])) == 0.

        @params LinearModel(;
                lift = (x) -> [[sin(x[1]), exp(x[2])]],
                theta_priors = fill(Dirac(1.), 2),
                noise_std_priors = fill(Dirac(0.1), 2),
            )
        @params LinearModel(;
                lift = (x) -> [[sin(x[1]), exp(x[2])]],
                theta_priors = fill(Dirac(1.), 2),
                noise_std_priors = fill(Dirac(0.1), 2),
            )
        @params NonlinearModel(;
                predict = (x, θ) -> [θ[1] * sin(x[1]) + θ[2] * exp(x[2])],
                theta_priors = fill(Dirac(1.), 2),
                noise_std_priors = fill(Dirac(0.1), 2),
            )
        @params NonlinearModel(;
                predict = (x, θ) -> [θ[1] * sin(x[1]) + θ[2] * exp(x[2])],
                theta_priors = fill(Dirac(1.), 2),
                noise_std_priors = fill(Dirac(0.1), 2),
            )
        @success (
            out(ParametricParams([1., 5.], [0.1, 0.1])) == -Inf,
            out(ParametricParams([1., 1.], [0.1, 0.5])) == -Inf,
        )
    end
end
