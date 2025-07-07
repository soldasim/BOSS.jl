
@testset "model_posterior(model, params, data)" begin
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = reduce(hcat, (x -> [sin(x[1]) + exp(x[2]), cos(x[1]) + exp(x[2])]).(eachcol(X)))
    
    problem = BossProblem(;
        f = x -> x,
        domain = Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        acquisition = ExpectedImprovement(;
            fitness = LinFitness([1., 0.]),
        ),
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
            all(var(out, [2., 2.]) .<= var(out, [3., 3.])),
            all(var(out, [10., 10.]) .<= var(out, [11., 11.])),

            # matrix
            mean(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractMatrix{<:Real},
            std(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractMatrix{<:Real},
            var(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractMatrix{<:Real},
            cov(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractArray{<:Real, 3},
            mean_and_std(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa Tuple{<:AbstractMatrix{<:Real}, <:AbstractMatrix{<:Real}},
            mean_and_var(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa Tuple{<:AbstractMatrix{<:Real}, <:AbstractMatrix{<:Real}},
            mean_and_cov(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa Tuple{<:AbstractMatrix{<:Real}, <:AbstractArray{<:Real, 3}},
            size(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (3, 2),
            size(std(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (3, 2),
            size(var(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (3, 2),
            size(cov(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (3, 3, 2),

            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_std(out, [1.;1.;; 2.;2.;; 3.;3.;;])[1]; atol=1e-8),
            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_var(out, [1.;1.;; 2.;2.;; 3.;3.;;])[1]; atol=1e-8),
            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_cov(out, [1.;1.;; 2.;2.;; 3.;3.;;])[1]; atol=1e-8),
            isapprox(std(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_std(out, [1.;1.;; 2.;2.;; 3.;3.;;])[2]; atol=1e-8),
            isapprox(var(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_var(out, [1.;1.;; 2.;2.;; 3.;3.;;])[2]; atol=1e-8),
            isapprox(cov(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_cov(out, [1.;1.;; 2.;2.;; 3.;3.;;])[2]; atol=1e-8),
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
            cov(out, [1.;1.;;]) isa AbstractArray{<:Real, 3},
            mean_and_std(out, [1.;1.;;]) isa Tuple{<:AbstractMatrix{<:Real}, <:AbstractMatrix{<:Real}},
            mean_and_var(out, [1.;1.;;]) isa Tuple{<:AbstractMatrix{<:Real}, <:AbstractMatrix{<:Real}},
            mean_and_cov(out, [1.;1.;;]) isa Tuple{<:AbstractMatrix{<:Real}, <:AbstractArray{<:Real, 3}},
            size(mean(out, [1.;1.;;])) == (1, 2),
            size(std(out, [1.;1.;;])) == (1, 2),
            size(var(out, [1.;1.;;])) == (1, 2),
            size(cov(out, [1.;1.;;])) == (1, 1, 2),

            isapprox(mean(out, [1.;1.;;]), mean_and_std(out, [1.;1.;;])[1]; atol=1e-8),
            isapprox(mean(out, [1.;1.;;]), mean_and_var(out, [1.;1.;;])[1]; atol=1e-8),
            isapprox(mean(out, [1.;1.;;]), mean_and_cov(out, [1.;1.;;])[1]; atol=1e-8),
            isapprox(std(out, [1.;1.;;]), mean_and_std(out, [1.;1.;;])[2]; atol=1e-8),
            isapprox(var(out, [1.;1.;;]), mean_and_var(out, [1.;1.;;])[2]; atol=1e-8),
            isapprox(cov(out, [1.;1.;;]), mean_and_cov(out, [1.;1.;;])[2]; atol=1e-8),
        )
    end
end

@testset "model_posterior_slice(model, params, data, slice)" begin
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = reduce(hcat, (x -> [sin(x[1]) + exp(x[2]), cos(x[1]) + exp(x[2])]).(eachcol(X)))
    
    problem = BossProblem(;
        f = x -> x,
        domain = Domain(; bounds=([0., 0.], [10., 10.])),
        y_max = [Inf, 5.],
        acquisition = ExpectedImprovement(;
            fitness = LinFitness([1., 0.]),
        ),
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
            var(out, [2., 2.]) < var(out, [3., 3.]),
            var(out, [10., 10.]) < var(out, [11., 11.]),

            # matrix
            mean(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractVector{<:Real},
            std(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractVector{<:Real},
            var(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractVector{<:Real},
            cov(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa AbstractMatrix{<:Real},
            mean_and_std(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            mean_and_var(out, [1.;1.;; 2.;2.;; 3.;3.;;]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            size(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (3,),
            size(std(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (3,),
            size(var(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (3,),
            size(cov(out, [1.;1.;; 2.;2.;; 3.;3.;;])) == (3, 3),

            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_std(out, [1.;1.;; 2.;2.;; 3.;3.;;])[1]; atol=1e-8),
            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_var(out, [1.;1.;; 2.;2.;; 3.;3.;;])[1]; atol=1e-8),
            isapprox(mean(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_cov(out, [1.;1.;; 2.;2.;; 3.;3.;;])[1]; atol=1e-8),
            isapprox(std(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_std(out, [1.;1.;; 2.;2.;; 3.;3.;;])[2]; atol=1e-8),
            isapprox(var(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_var(out, [1.;1.;; 2.;2.;; 3.;3.;;])[2]; atol=1e-8),
            isapprox(cov(out, [1.;1.;; 2.;2.;; 3.;3.;;]), mean_and_cov(out, [1.;1.;; 2.;2.;; 3.;3.;;])[2]; atol=1e-8),
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
            cov(out, [1.;1.;;]) isa AbstractMatrix{<:Real},
            mean_and_std(out, [1.;1.;;]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            mean_and_var(out, [1.;1.;;]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            mean_and_cov(out, [1.;1.;;]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractMatrix{<:Real}},
            size(mean(out, [1.;1.;;])) == (1,),
            size(std(out, [1.;1.;;])) == (1,),
            size(var(out, [1.;1.;;])) == (1,),
            size(cov(out, [1.;1.;;])) == (1, 1),

            isapprox(mean(out, [1.;1.;;]), mean_and_std(out, [1.;1.;;])[1]; atol=1e-8),
            isapprox(mean(out, [1.;1.;;]), mean_and_var(out, [1.;1.;;])[1]; atol=1e-8),
            isapprox(mean(out, [1.;1.;;]), mean_and_cov(out, [1.;1.;;])[1]; atol=1e-8),
            isapprox(std(out, [1.;1.;;]), mean_and_std(out, [1.;1.;;])[2]; atol=1e-8),
            isapprox(var(out, [1.;1.;;]), mean_and_var(out, [1.;1.;;])[2]; atol=1e-8),
            isapprox(cov(out, [1.;1.;;]), mean_and_cov(out, [1.;1.;;])[2]; atol=1e-8),
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
