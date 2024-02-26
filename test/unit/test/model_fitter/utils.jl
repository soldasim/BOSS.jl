
@testset "sample_params(model, noise_var_priors)" begin
    @param_test BOSS.sample_params begin
        @params (
            BOSS.LinModel(;
                lift = (x) -> [
                    [sin(x[1]), exp(x[2])],
                    [cos(x[1]), exp(x[2])],
                ],
                param_priors = fill(BOSS.Normal(), 4),
            ),
            fill(BOSS.LogNormal(), 2),
        )
        @success (
            length(out) == 3,
            out[1] isa AbstractVector{<:Real},
            out[2] isa AbstractMatrix{<:Real},
            out[3] isa AbstractVector{<:Real},
            length(out[1]) == 4,
            isempty(out[2]),
            length(out[3]) == 2,
        )

        @params (
            BOSS.NonlinModel(;
                predict = x -> [
                    θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
                    θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
                ],
                param_priors = fill(BOSS.Normal(), 4),
            ),
            fill(BOSS.LogNormal(), 2),
        )
        @success (
            length(out) == 3,
            out[1] isa AbstractVector{<:Real},
            out[2] isa AbstractMatrix{<:Real},
            out[3] isa AbstractVector{<:Real},
            length(out[1]) == 4,
            isempty(out[2]),
            length(out[3]) == 2,
        )

        @params (
            BOSS.Nonparametric(;
                length_scale_priors = fill(BOSS.MvLogNormal(ones(2), ones(2)), 2),
            ),
            fill(BOSS.LogNormal(), 2),
        )
        @success (
            length(out) == 3,
            out[1] isa AbstractVector{<:Real},
            out[2] isa AbstractMatrix{<:Real},
            out[3] isa AbstractVector{<:Real},
            isempty(out[1]),
            size(out[2]) == (2, 2),
            length(out[3]) == 2,
        )

        @params (
            BOSS.Semiparametric(
                parametric = BOSS.NonlinModel(;
                    predict = x -> [
                        θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
                        θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
                    ],
                    param_priors = fill(BOSS.Normal(), 4),
                ),
                nonparametric = BOSS.Nonparametric(;
                    length_scale_priors = fill(BOSS.MvLogNormal(ones(2), ones(2)), 2),
                ),
            ),
            fill(BOSS.LogNormal(), 2),
        )
        @success (
            length(out) == 3,
            out[1] isa AbstractVector{<:Real},
            out[2] isa AbstractMatrix{<:Real},
            out[3] isa AbstractVector{<:Real},
            length(out[1]) == 4,
            size(out[2]) == (2, 2),
            length(out[3]) == 2,
        )
    end
end

@testset "vectorize_params(θ, λ, noise_vars, activation_function, activation_mask, skip_mask)" begin
    activation_function(x) = -x
    BOSS.inverse(activation_function) = activation_function
    
    @param_test BOSS.vectorize_params begin
        @params [1., 2., 3.], [4.;4.;; 5.;5.;;], [0.1, 0.1], activation_function, fill(false, 9), fill(true, 9)
        @success out == [1., 2., 3., 4., 4., 5., 5., 0.1, 0.1]

        @params [1., 2., 3.], [4.;4.;; 5.;5.;;], [0.1, 0.1], activation_function, fill(true, 9), fill(true, 9)
        @success out == -1. * [1., 2., 3., 4., 4., 5., 5., 0.1, 0.1]

        @params [1., 2., 3.], [4.;4.;; 5.;5.;;], [0.1, 0.1], activation_function, vcat([true, false, true], fill(false, 6)), fill(true, 9)
        @success out == [-1., 2., -3., 4., 4., 5., 5., 0.1, 0.1]

        @params [1., 2., 3.], [4.;4.;; 5.;5.;;], [0.1, 0.1], activation_function, fill(false, 9), vcat(fill(true, 3), fill(false, 6))
        @success out == [1., 2., 3.]

        @params [1., 2., 3.], [4.;4.;; 5.;5.;;], [0.1, 0.1], activation_function, vcat([true, false, true], fill(false, 6)), vcat(fill(true, 3), fill(false, 6))
        @success out == [-1., 2., -3.]

        @params [1., 2., 3.], [4.;4.;; 5.;5.;;], [0.1, 0.1], activation_function, vcat([true, false, true], fill(false, 6)), vcat([true, false, true], fill(true, 6))
        @success out == [-1., -3., 4., 4., 5., 5., 0.1, 0.1]

        @params [1., 2., 3.], [4.;4.;; 5.;5.;;], [0.1, 0.1], activation_function, fill(false, 9), fill(false, 9)
        @params [1., 2., 3.], [4.;4.;; 5.;5.;;], [0.1, 0.1], activation_function, fill(true, 9), fill(false, 9)
        @success out == Float64[]
    end
end

@testset "vectorize_params(θ, λ, noise_vars)" begin
    @param_test BOSS.vectorize_params begin
        @params [1., 2., 3.], [4.;4.;; 5.;5.;;], [0.1, 0.1]
        @success (
            out == [1., 2., 3., 4., 4., 5., 5., 0.1, 0.1],
            eltype(out) == Float64,
        )

        @params Real[], [4.;4.;; 5.;5.;;], [0.1, 0.1]
        @success (
            out == [4., 4., 5., 5., 0.1, 0.1],
            eltype(out) == Float64,
        )

        @params [1., 2., 3.], Float64[;;], [0.1, 0.1]
        @success (
            out == [1., 2., 3., 0.1, 0.1],
            eltype(out) == Float64,
        )

        @params [1., 2., 3.], [4.;4.;; 5.;5.;;], Float64[]
        @success (
            out == [1., 2., 3., 4., 4., 5., 5.],
            eltype(out) == Float64,
        )

        @params Real[], Real[;;], Real[]
        @success (
            out == Real[],
            eltype(out) == Real,
        )
    end
end

@testset "devectorize_params(model, params, activation_function, activation_mask, skipped_values, skip_mask)" begin
    lin_model = BOSS.LinModel(;
        lift = (x) -> [
            [sin(x[1]), exp(x[2])],
            [cos(x[1]), exp(x[2])],
        ],
        param_priors = fill(BOSS.Normal(), 4),
    )
    nonlin_model = BOSS.NonlinModel(;
        predict = (x, θ) -> [
            θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
            θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
        ],
        param_priors = fill(BOSS.Normal(), 4),
    )
    nonparametric = BOSS.Nonparametric(;
        kernel = BOSS.Matern52Kernel(),
        length_scale_priors = fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2),
    )
    semiparametric = BOSS.Semiparametric(;
        parametric = nonlin_model,
        nonparametric = nonparametric,
    )

    activation_function(x) = -x
    BOSS.inverse(activation_function) = activation_function    

    @param_test BOSS.devectorize_params begin
        @params deepcopy(lin_model), [1., 2., 3., 4., 0.1, 0.1], activation_function, fill(false, 6), Float64[], fill(true, 6)
        @params deepcopy(nonlin_model), [1., 2., 3., 4., 0.1, 0.1], activation_function, fill(false, 6), Float64[], fill(true, 6)
        @success out == ([1., 2., 3., 4.], Float64[;;], [0.1, 0.1])

        @params deepcopy(nonparametric), [4., 4., 5., 5., 0.1, 0.1], activation_function, fill(false, 6), Float64[], fill(true, 6)
        @success out == (Float64[], [4.;4.;; 5.;5.;;], [0.1, 0.1])

        @params deepcopy(semiparametric), [1., 2., 3., 4., 4., 4., 5., 5., 0.1, 0.1], activation_function, fill(false, 10), Float64[], fill(true, 10)
        @success out == ([1., 2., 3., 4.], [4.;4.;; 5.;5.;;], [0.1, 0.1])

        @params deepcopy(lin_model), [-1., 2., -3., 4., 0.1, 0.1], activation_function, vcat([true, false, true, false], fill(false, 2)), Float64[], fill(true, 6)
        @params deepcopy(nonlin_model), [-1., 2., -3., 4., 0.1, 0.1], activation_function, vcat([true, false, true, false], fill(false, 2)), Float64[], fill(true, 6)
        @success out == ([1., 2., 3., 4.], Float64[;;], [0.1, 0.1])

        @params deepcopy(nonparametric), [-4., -4., 5., 5., 0.1, 0.1], activation_function, vcat([true, true, false, false], fill(false, 2)), Float64[], fill(true, 6)
        @success out == (Float64[], [4.;4.;; 5.;5.;;], [0.1, 0.1])

        @params deepcopy(semiparametric), [-1., 2., -3., 4., 4., 4., 5., 5., 0.1, 0.1], activation_function, vcat([true, false, true, false], fill(false, 6)), Float64[], fill(true, 10)
        @success out == ([1., 2., 3., 4.], [4.;4.;; 5.;5.;;], [0.1, 0.1])

        @params deepcopy(lin_model), [2., 4., 0.1, 0.1], activation_function, fill(false, 6), [1., 3.], vcat([false, true, false, true], fill(true, 2))
        @params deepcopy(nonlin_model), [2., 4., 0.1, 0.1], activation_function, fill(false, 6), [1., 3.], vcat([false, true, false, true], fill(true, 2))
        @success out == ([1., 2., 3., 4.], Float64[;;], [0.1, 0.1])

        @params deepcopy(nonparametric), [5., 5., 0.1, 0.1], activation_function, fill(false, 6), [4., 4.], vcat([false, false, true, true], fill(true, 2))
        @success out == (Float64[], [4.;4.;; 5.;5.;;], [0.1, 0.1])

        @params deepcopy(semiparametric), [2., 4., 4., 4., 5., 5., 0.1, 0.1], activation_function, fill(false, 10), [1., 3.], vcat([false, true, false, true], fill(true, 6))
        @success out == ([1., 2., 3., 4.], [4.;4.;; 5.;5.;;], [0.1, 0.1])

        @params deepcopy(lin_model), [-2., -4., 0.1, 0.1], activation_function, vcat([false, true, false, true], fill(false, 2)), [1., 3.], vcat([false, true, false, true], fill(true, 2))
        @params deepcopy(nonlin_model), [-2., -4., 0.1, 0.1], activation_function, vcat([false, true, false, true], fill(false, 2)), [1., 3.], vcat([false, true, false, true], fill(true, 2))
        @success out == ([1., 2., 3., 4.], Float64[;;], [0.1, 0.1])

        @params deepcopy(nonparametric), [-5., -5., 0.1, 0.1], activation_function, vcat([false, false, true, true], fill(false, 2)), [4., 4.], vcat([false, false, true, true], fill(true, 2))
        @success out == (Float64[], [4.;4.;; 5.;5.;;], [0.1, 0.1])

        @params deepcopy(semiparametric), [-2., -4., 4., 4., 5., 5., 0.1, 0.1], activation_function, vcat([false, true, false, true], fill(false, 6)), [1., 3.], vcat([false, true, false, true], fill(true, 6))
        @success out == ([1., 2., 3., 4.], [4.;4.;; 5.;5.;;], [0.1, 0.1])
    end
end

@testset "devectorize_params(model, params)" begin
    lin_model = BOSS.LinModel(;
        lift = (x) -> [
            [sin(x[1]), exp(x[2])],
            [cos(x[1]), exp(x[2])],
        ],
        param_priors = fill(BOSS.Normal(), 4),
    )
    nonlin_model = BOSS.NonlinModel(;
        predict = (x, θ) -> [
            θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
            θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
        ],
        param_priors = fill(BOSS.Normal(), 4),
    )
    nonparametric = BOSS.Nonparametric(;
        kernel = BOSS.Matern52Kernel(),
        length_scale_priors = fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2),
    )
    semiparametric = BOSS.Semiparametric(;
        parametric = nonlin_model,
        nonparametric = nonparametric,
    )

    @param_test BOSS.devectorize_params begin
        @params deepcopy(lin_model), [1., 2., 3., 4., 0.1, 0.1]
        @params deepcopy(nonlin_model), [1., 2., 3., 4., 0.1, 0.1]
        @success out == ([1., 2., 3., 4.], Float64[;;], [0.1, 0.1])

        @params deepcopy(nonparametric), [4., 4., 5., 5., 0.1, 0.1]
        @success out == (Float64[], [4.;4.;; 5.;5.;;], [0.1, 0.1])

        @params deepcopy(semiparametric), [1., 2., 3., 4., 4., 4., 5., 5., 0.1, 0.1]
        @success out == ([1., 2., 3., 4.], [4.;4.;; 5.;5.;;], [0.1, 0.1])
    end
end

@testset "create_activation_mask(params_total, θ_len, mask_noisevar_and_lengthscales, mask_theta)" begin
    @param_test BOSS.create_activation_mask begin
        @params 6, 4, false, nothing
        @success out == fill(false, 6)

        @params 6, 4, true, nothing
        @success out == vcat(fill(false, 4), fill(true, 2))

        @params 6, 4, false, [true, false, true, false]
        @success out == vcat([true, false, true, false], fill(false, 2))

        @params 6, 0, false, nothing
        @params 6, 0, false, Bool[]
        @success out == fill(false, 6)

        @params 6, 0, true, nothing
        @success out == fill(true, 6)

        @params 10, 4, false, nothing
        @success out == fill(false, 10)

        @params 10, 4, true, nothing
        @success out == vcat(fill(false, 4), fill(true, 6))

        @params 10, 4, false, [true, false, true, false]
        @success out == vcat([true, false, true, false], fill(false, 6))

        @params 10, 4, true, [true, false, true, false]
        @success out == vcat([true, false, true, false], fill(true, 6))
    end
end

@testset "create_dirac_skip_mask(θ_priors, λ_priors, noise_var_priors)" begin
    @param_test BOSS.create_dirac_skip_mask begin
        @params fill(BOSS.Normal(), 4), BOSS.MultivariateDistribution[], fill(BOSS.LogNormal(), 2)
        @success out == (fill(true, 6), Float64[])

        @params [BOSS.Dirac(1.), BOSS.Normal(), BOSS.Dirac(1.), BOSS.Normal()], BOSS.MultivariateDistribution[], fill(BOSS.LogNormal(), 2)
        @success out == (vcat([false, true, false, true], fill(true, 2)), [1., 1.])

        @params fill(BOSS.Normal(), 4), BOSS.MultivariateDistribution[], fill(BOSS.Dirac(0.1), 2)
        @success out == (vcat(fill(true, 4), [false, false]), [0.1, 0.1])

        @params BOSS.UnivariateDistribution[], fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2), fill(BOSS.LogNormal(), 2)
        @success out == (fill(true, 6), Float64[])

        @params BOSS.UnivariateDistribution[], [BOSS.Product(fill(BOSS.Dirac(1.), 2)), BOSS.MvLogNormal([1., 1.], [1., 1.])], fill(BOSS.LogNormal(), 2)
        @success out == (vcat([false, false, true, true], fill(true,  2)), [1., 1.])

        @params BOSS.UnivariateDistribution[], fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2), fill(BOSS.Dirac(0.1), 2)
        @success out == (vcat(fill(true,4), [false, false]), [0.1, 0.1])

        @params fill(BOSS.Normal(), 4), fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2), fill(BOSS.LogNormal(), 2)
        @success out == (fill(true, 10), Float64[])

        @params [BOSS.Dirac(1.), BOSS.Normal(), BOSS.Dirac(1.), BOSS.Normal()], fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2), fill(BOSS.LogNormal(), 2)
        @success out == (vcat([false, true, false, true], fill(true, 6)), [1., 1.])

        @params fill(BOSS.Normal(), 4), [BOSS.Product(fill(BOSS.Dirac(1.), 2)), BOSS.MvLogNormal([1., 1.], [1., 1.])], fill(BOSS.LogNormal(), 2)
        @success out == (vcat(fill(true, 4), [false, false, true, true], fill(true, 2)), [1., 1.])

        @params fill(BOSS.Normal(), 4), fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2), fill(BOSS.Dirac(0.1), 2)
        @success out == (vcat(fill(true, 8), [false, false]), [0.1, 0.1])

        @params [BOSS.Dirac(1.), BOSS.Normal(), BOSS.Dirac(1.), BOSS.Normal()], fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2), fill(BOSS.Dirac(0.1), 2)
        @success out == (vcat([false, true, false, true], fill(true, 4), [false, false]), [1., 1., 0.1, 0.1])
    end
end
