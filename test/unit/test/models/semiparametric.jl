
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
                param_priors = fill(BOSS.Normal(), 4),
            ),
            nonparametric = BOSS.Nonparametric(;
                amp_priors = fill(BOSS.LogNormal(), 2),
                length_scale_priors = fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2),
            ),
        ),
        noise_var_priors = fill(BOSS.Dirac(1e-8), 2),
        data = BOSS.ExperimentDataPrior(X, Y),
    )
    BOSS.estimate_parameters!(problem, BOSS.SamplingMLE(; samples=200, parallel=PARALLEL_TESTS); options=BOSS.BossOptions(; info=false))

    @param_test (m, d) -> BOSS.model_posterior(m, d; split=false) begin
        @params problem.model, problem.data
        @success (
            out isa Function,
            out([2., 2.]) isa Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}},
            length(out([2., 2.])[1]) == 2,
            length(out([2., 2.])[2]) == 2,
            all(out([2., 2.])[2] .< out([3., 3.])[2]),
            all(out([10., 10.])[2] .< out([11., 11.])[2]),
        )
    end
    @param_test (m, d) -> BOSS.model_posterior(m, d; split=true) begin
        @params problem.model, problem.data
        @success (
            out isa Vector,
            out[1]([2., 2.]) isa Tuple{<:Real, <:Real},
            out[1]([2., 2.])[2] < out[1]([3., 3.])[2],
            out[2]([2., 2.])[2] < out[2]([3., 3.])[2],
            out[1]([10., 10.])[2] < out[1]([11., 11.])[2],
            out[2]([10., 10.])[2] < out[2]([11., 11.])[2],
        )
    end
end

@testset "model_loglike(model, noise_var_priors, data)" begin
    X = [2.;2.;; 5.;5.;; 8.;8.;;]
    Y = reduce(hcat, (x -> [sin(x[1]) + exp(x[2]), cos(x[1]) + exp(x[2])]).(eachcol(X)))
    
    model = BOSS.Semiparametric(;
        parametric = BOSS.NonlinModel(;
            predict = (x, θ) -> [
                θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
                θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
            ],
            param_priors = fill(BOSS.Normal(), 4),
        ),
        nonparametric = BOSS.Nonparametric(;
        amp_priors = fill(BOSS.LogNormal(), 2),
            length_scale_priors = fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2),
        ),
    )
    noise_var_priors = fill(BOSS.LogNormal(1., 1.), 2)
    data = BOSS.ExperimentDataPrior(X, Y)

    @param_test BOSS.model_loglike begin
        @params deepcopy(model), deepcopy(noise_var_priors), deepcopy(data)
        @success (
            out isa Function,
            out([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.]) isa Real,
            out([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.]) < 0.,
            out([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [5., 5.]) > out([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [100., 100.]),
            out([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [5., 5.], [1., 1.]) > out([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [100., 100.], [1., 1.]),
            out([1., 1., 1., 1.], [5.;5.;; 5.;5.;;], [1., 1.], [1., 1.]) > out([1., 1., 1., 1.], [100.;100.;; 100.;100.;;], [1., 1.], [1., 1.]),
            out([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.]) > out([10., 10., 10., 10.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.]),
            out([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.]) > out([2., 2., 2., 2.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.]),
            out([1., 1., 1., 1.], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.]) > out([0.5, 0.5, 0.5, 0.5], [1.;1.;; 1.;1.;;], [1., 1.], [1., 1.]),
        )
    end
end
