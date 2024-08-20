
@testset "sample_params(model, noise_std_priors)" begin
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
            length(out) == 4,
            out[1] isa AbstractVector{<:Real},
            out[2] isa AbstractMatrix{<:Real},
            out[3] isa AbstractVector{<:Real},
            out[4] isa AbstractVector{<:Real},
            size(out[1]) == (4,),
            isempty(out[2]),
            isempty(out[3]),
            size(out[4]) == (2,),
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
            length(out) == 4,
            out[1] isa AbstractVector{<:Real},
            out[2] isa AbstractMatrix{<:Real},
            out[3] isa AbstractVector{<:Real},
            out[4] isa AbstractVector{<:Real},
            size(out[1]) == (4,),
            isempty(out[2]),
            isempty(out[3]),
            size(out[4]) == (2,),
        )

        @params (
            BOSS.Nonparametric(;
                amp_priors = fill(BOSS.LogNormal(), 2),
                length_scale_priors = fill(BOSS.MvLogNormal(ones(2), ones(2)), 2),
            ),
            fill(BOSS.LogNormal(), 2),
        )
        @success (
            length(out) == 4,
            out[1] isa AbstractVector{<:Real},
            out[2] isa AbstractMatrix{<:Real},
            out[3] isa AbstractVector{<:Real},
            out[4] isa AbstractVector{<:Real},
            isempty(out[1]),
            size(out[2]) == (2, 2),
            size(out[3]) == (2,),
            size(out[4]) == (2,),
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
                    amp_priors = fill(BOSS.LogNormal(), 2),
                    length_scale_priors = fill(BOSS.MvLogNormal(ones(2), ones(2)), 2),
                ),
            ),
            fill(BOSS.LogNormal(), 2),
        )
        @success (
            length(out) == 4,
            out[1] isa AbstractVector{<:Real},
            out[2] isa AbstractMatrix{<:Real},
            out[3] isa AbstractVector{<:Real},
            out[4] isa AbstractVector{<:Real},
            size(out[1]) == (4,),
            size(out[2]) == (2, 2),
            size(out[3]) == (2,),
            size(out[4]) == (2,),
        )
    end
end

@testset "vectorize_params(params, activation_function, activation_mask, skip_mask)" begin
    activation_function(x) = -x
    BOSS.inverse(activation_function) = activation_function
    
    @param_test BOSS.vectorize_params begin
        @params ([1., 2., 3.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1]), activation_function, fill(false, 11), fill(true, 11)
        @success out == [1., 2., 3., 4., 4., 5., 5., 1., 1., 0.1, 0.1]

        @params ([1., 2., 3.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1]), activation_function, fill(true, 11), fill(true, 11)
        @success out == -1. * [1., 2., 3., 4., 4., 5., 5., 1., 1., 0.1, 0.1]

        @params ([1., 2., 3.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1]), activation_function, vcat([true, false, true], fill(false, 8)), fill(true, 11)
        @success out == [-1., 2., -3., 4., 4., 5., 5., 1., 1., 0.1, 0.1]

        @params ([1., 2., 3.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1]), activation_function, fill(false, 11), vcat(fill(true, 3), fill(false, 8))
        @success out == [1., 2., 3.]

        @params ([1., 2., 3.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1]), activation_function, vcat([true, false, true], fill(false, 8)), vcat(fill(true, 3), fill(false, 8))
        @success out == [-1., 2., -3.]

        @params ([1., 2., 3.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1]), activation_function, vcat([true, false, true], fill(false, 8)), vcat([true, false, true], fill(true, 8))
        @success out == [-1., -3., 4., 4., 5., 5., 1., 1., 0.1, 0.1]

        @params ([1., 2., 3.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1]), activation_function, fill(false, 11), fill(false, 11)
        @params ([1., 2., 3.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1]), activation_function, fill(true, 11), fill(false, 11)
        @success out == Float64[]
    end
end

@testset "vectorize_params(θ, λ, α, noise_std)" begin
    @param_test BOSS.vectorize_params begin
        @params [1., 2., 3.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1]
        @success (
            out == [1., 2., 3., 4., 4., 5., 5., 1., 1., 0.1, 0.1],
            eltype(out) == Float64,
        )

        @params Real[], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1]
        @success (
            out == [4., 4., 5., 5., 1., 1., 0.1, 0.1],
            eltype(out) == Float64,
        )

        @params [1., 2., 3.], Float64[;;], [1., 1.], [0.1, 0.1]
        @success (
            out == [1., 2., 3., 1., 1., 0.1, 0.1],
            eltype(out) == Float64,
        )

        @params [1., 2., 3.], [4.;4.;; 5.;5.;;], Float64[], [0.1, 0.1]
        @success (
            out == [1., 2., 3., 4., 4., 5., 5., 0.1, 0.1],
            eltype(out) == Float64,
        )

        @params [1., 2., 3.], [4.;4.;; 5.;5.;;], [1., 1.], Float64[]
        @success (
            out == [1., 2., 3., 4., 4., 5., 5., 1., 1.],
            eltype(out) == Float64,
        )

        @params Real[], Real[;;], Real[], Real[]
        @success (
            out == Real[],
            eltype(out) == Real,
        )

        @params Float64[], Float64[;;], Float64[], Float64[]
        @success (
            out == Float64[],
            eltype(out) == Float64,
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
        amp_priors = fill(BOSS.LogNormal(), 2),
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
        @success out == ([1., 2., 3., 4.], Float64[;;], Float64[], [0.1, 0.1])

        @params deepcopy(nonparametric), [4., 4., 5., 5., 1., 1., 0.1, 0.1], activation_function, fill(false, 8), Float64[], fill(true, 8)
        @success out == (Float64[], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])

        @params deepcopy(semiparametric), [1., 2., 3., 4., 4., 4., 5., 5., 1., 1., 0.1, 0.1], activation_function, fill(false, 12), Float64[], fill(true, 12)
        @success out == ([1., 2., 3., 4.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])

        @params deepcopy(lin_model), [-1., 2., -3., 4., 0.1, 0.1], activation_function, vcat([true, false, true, false], fill(false, 2)), Float64[], fill(true, 6)
        @params deepcopy(nonlin_model), [-1., 2., -3., 4., 0.1, 0.1], activation_function, vcat([true, false, true, false], fill(false, 2)), Float64[], fill(true, 6)
        @success out == ([1., 2., 3., 4.], Float64[;;], Float64[], [0.1, 0.1])

        @params deepcopy(nonparametric), [-4., -4., -5., -5., -1., -1., -0.1, -0.1], activation_function, fill(true, 8), Float64[], fill(true, 8)
        @success out == (Float64[], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])

        @params deepcopy(semiparametric), [-1., 2., -3., 4., 4., 4., 5., 5., 1., 1., 0.1, 0.1], activation_function, vcat([true, false, true, false], fill(false, 8)), Float64[], fill(true, 12)
        @success out == ([1., 2., 3., 4.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])

        @params deepcopy(lin_model), [2., 4., 0.1, 0.1], activation_function, fill(false, 6), [1., 3.], vcat([false, true, false, true], fill(true, 2))
        @params deepcopy(nonlin_model), [2., 4., 0.1, 0.1], activation_function, fill(false, 6), [1., 3.], vcat([false, true, false, true], fill(true, 2))
        @success out == ([1., 2., 3., 4.], Float64[;;], Float64[], [0.1, 0.1])

        @params deepcopy(nonparametric), [5., 5., 1., 0.1, 0.1], activation_function, fill(false, 8), [4., 4., 1.], vcat([false, false, true, true], [false, true], fill(true, 2))
        @success out == (Float64[], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])

        @params deepcopy(semiparametric), [2., 4., 4., 4., 5., 5., 1., 1., 0.1, 0.1], activation_function, fill(false, 12), [1., 3.], vcat([false, true, false, true], fill(true, 8))
        @success out == ([1., 2., 3., 4.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])

        @params deepcopy(lin_model), [3., -4., 0.1, 0.1], activation_function, vcat([false, true, false, true], fill(false, 2)), [1., 2.], vcat([false, false, true, true], fill(true, 2))
        @params deepcopy(nonlin_model), [3., -4., 0.1, 0.1], activation_function, vcat([false, true, false, true], fill(false, 2)), [1., 2.], vcat([false, false, true, true], fill(true, 2))
        @success out == ([1., 2., 3., 4.], Float64[;;], Float64[], [0.1, 0.1])

        @params deepcopy(nonparametric), [-5., -5., -1., -0.1, -0.1], activation_function, fill(true, 8), [4., 4., 1.], vcat([false, false, true, true], [false, true], fill(true, 2))
        @success out == (Float64[], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])

        @params deepcopy(semiparametric), [3., -4., 4., 4., 5., 5., 1., 1., 0.1, 0.1], activation_function, vcat([false, true, false, true], fill(false, 8)), [1., 2.], vcat([false, false, true, true], fill(true, 8))
        @success out == ([1., 2., 3., 4.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])
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
        amp_priors = fill(BOSS.LogNormal(), 2),
        length_scale_priors = fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2),
    )
    semiparametric = BOSS.Semiparametric(;
        parametric = nonlin_model,
        nonparametric = nonparametric,
    )

    @param_test BOSS.devectorize_params begin
        @params deepcopy(lin_model), [1., 2., 3., 4., 0.1, 0.1]
        @params deepcopy(nonlin_model), [1., 2., 3., 4., 0.1, 0.1]
        @success out == ([1., 2., 3., 4.], Float64[;;], Float64[], [0.1, 0.1])

        @params deepcopy(nonparametric), [4., 4., 5., 5., 1., 1., 0.1, 0.1]
        @success out == (Float64[], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])

        @params deepcopy(semiparametric), [1., 2., 3., 4., 4., 4., 5., 5., 1., 1., 0.1, 0.1]
        @success out == ([1., 2., 3., 4.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])
    end
end

@testset "create_activation_mask(params_total, θ_len, mask_hyperparams, mask_params)" begin
    @param_test BOSS.create_activation_mask begin
        @params 6, 4, false, false
        @params 6, 4, false, fill(false, 4)
        @success out == fill(false, 6)

        @params 6, 4, true, false
        @params 6, 4, true, fill(false, 4)
        @success out == vcat(fill(false, 4), fill(true, 2))

        @params 6, 4, false, true
        @params 6, 4, false, fill(true, 4)
        @success out == vcat(fill(true, 4), fill(false, 2))

        @params 6, 4, false, [true, false, true, false]
        @success out == vcat([true, false, true, false], fill(false, 2))

        @params 8, 0, false, false
        @params 8, 0, false, true
        @params 8, 0, false, Bool[]
        @success out == fill(false, 8)

        @params 8, 0, true, false
        @params 8, 0, true, true
        @params 8, 0, true, Bool[]
        @success out == fill(true, 8)

        @params 12, 4, false, false
        @params 12, 4, false, fill(false, 4)
        @success out == fill(false, 12)

        @params 12, 4, true, false
        @params 12, 4, true, fill(false, 4)
        @success out == vcat(fill(false, 4), fill(true, 8))

        @params 12, 4, false, [true, false, true, false]
        @success out == vcat([true, false, true, false], fill(false, 8))

        @params 12, 4, true, [true, false, true, false]
        @success out == vcat([true, false, true, false], fill(true, 8))
    end
end

@testset "create_dirac_skip_mask(θ_priors, λ_priors, α_priors, noise_std_priors)" begin
    @param_test BOSS.create_dirac_skip_mask begin
        @params fill(BOSS.Normal(), 4), BOSS.MultivariateDistribution[], BOSS.UnivariateDistribution[], fill(BOSS.LogNormal(), 2)
        @success out == (fill(true, 6), Float64[])

        @params [BOSS.Dirac(1.), BOSS.Normal(), BOSS.Dirac(3.), BOSS.Normal()], BOSS.MultivariateDistribution[], BOSS.UnivariateDistribution[], fill(BOSS.LogNormal(), 2)
        @success out == (vcat([false, true, false, true], fill(true, 2)), [1., 3.])

        @params fill(BOSS.Normal(), 4), BOSS.MultivariateDistribution[], BOSS.UnivariateDistribution[], fill(BOSS.Dirac(0.1), 2)
        @success out == (vcat(fill(true, 4), [false, false]), [0.1, 0.1])

        @params BOSS.UnivariateDistribution[], fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2), fill(BOSS.LogNormal(), 2), fill(BOSS.LogNormal(), 2)
        @success out == (fill(true, 8), Float64[])

        @params BOSS.UnivariateDistribution[], [BOSS.Product(fill(BOSS.Dirac(1.), 2)), BOSS.MvLogNormal([1., 1.], [1., 1.])], [BOSS.Dirac(1.), BOSS.LogNormal()], fill(BOSS.LogNormal(), 2)
        @success out == (vcat([false, false, true, true], [false, true], fill(true,  2)), [1., 1., 1.])

        @params BOSS.UnivariateDistribution[], fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2), fill(BOSS.LogNormal(), 2), fill(BOSS.Dirac(0.1), 2)
        @success out == (vcat(fill(true, 6), fill(false, 2)), [0.1, 0.1])

        @params fill(BOSS.Normal(), 4), fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2), fill(BOSS.LogNormal(), 2), fill(BOSS.LogNormal(), 2)
        @success out == (fill(true, 12), Float64[])

        @params [BOSS.Dirac(1.), BOSS.Normal(), BOSS.Dirac(3.), BOSS.Normal()], fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2), fill(BOSS.LogNormal(), 2), fill(BOSS.LogNormal(), 2)
        @success out == (vcat([false, true, false, true], fill(true, 8)), [1., 3.])

        @params fill(BOSS.Normal(), 4), [BOSS.Product(fill(BOSS.Dirac(1.), 2)), BOSS.MvLogNormal([1., 1.], [1., 1.])], [BOSS.Dirac(1.), BOSS.LogNormal()], fill(BOSS.LogNormal(), 2)
        @success out == (vcat(fill(true, 4), [false, false, true, true], [false, true], fill(true, 2)), [1., 1., 1.])

        @params fill(BOSS.Normal(), 4), fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2), fill(BOSS.LogNormal(), 2), fill(BOSS.Dirac(0.1), 2)
        @success out == (vcat(fill(true, 10), [false, false]), [0.1, 0.1])

        @params [BOSS.Dirac(1.), BOSS.Normal(), BOSS.Dirac(3.), BOSS.Normal()], fill(BOSS.MvLogNormal([1., 1.], [1., 1.]), 2), fill(BOSS.LogNormal(), 2), fill(BOSS.Dirac(0.1), 2)
        @success out == (vcat([false, true, false, true], fill(true, 4), fill(true, 2), [false, false]), [1., 3., 0.1, 0.1])
    end
end
