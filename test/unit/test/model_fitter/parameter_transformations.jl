
@testset "vectorize_params(params, activation_function, activation_mask, skip_mask)" begin
    activation_function(x) = -x
    BOSS.inverse(::typeof(activation_function)) = activation_function
    
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

@testset "devectorize_params(params, model, activation_function, activation_mask, skipped_values, skip_mask)" begin
    lin_model = LinModel(;
        lift = (x) -> [
            [sin(x[1]), exp(x[2])],
            [cos(x[1]), exp(x[2])],
        ],
        theta_priors = fill(Normal(), 4),
        noise_std_priors = fill(Dirac(0.1), 2),
    )
    nonlin_model = NonlinModel(;
        predict = (x, θ) -> [
            θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
            θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
        ],
        theta_priors = fill(Normal(), 4),
        noise_std_priors = fill(Dirac(0.1), 2),
    )
    nonparametric = Nonparametric(;
        kernel = Matern32Kernel(),
        amp_priors = fill(LogNormal(), 2),
        length_scale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
        noise_std_priors = fill(Dirac(0.1), 2),
    )
    semiparametric = Semiparametric(;
        parametric = nonlin_model,
        nonparametric = nonparametric,
    )

    activation_function(x) = -x
    BOSS.inverse(::typeof(activation_function)) = activation_function    

    @param_test BOSS.devectorize_params begin
        @params [1., 2., 3., 4., 0.1, 0.1], deepcopy(lin_model), activation_function, fill(false, 6), Float64[], fill(true, 6)
        @params [1., 2., 3., 4., 0.1, 0.1], deepcopy(nonlin_model), activation_function, fill(false, 6), Float64[], fill(true, 6)
        @success out == ([1., 2., 3., 4.], nothing, nothing, [0.1, 0.1])

        @params [4., 4., 5., 5., 1., 1., 0.1, 0.1], deepcopy(nonparametric), activation_function, fill(false, 8), Float64[], fill(true, 8)
        @success out == (nothing, [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])

        @params [1., 2., 3., 4., 4., 4., 5., 5., 1., 1., 0.1, 0.1], deepcopy(semiparametric), activation_function, fill(false, 12), Float64[], fill(true, 12)
        @success out == ([1., 2., 3., 4.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])

        @params [-1., 2., -3., 4., 0.1, 0.1], deepcopy(lin_model), activation_function, vcat([true, false, true, false], fill(false, 2)), Float64[], fill(true, 6)
        @params [-1., 2., -3., 4., 0.1, 0.1], deepcopy(nonlin_model), activation_function, vcat([true, false, true, false], fill(false, 2)), Float64[], fill(true, 6)
        @success out == ([1., 2., 3., 4.], nothing, nothing, [0.1, 0.1])

        @params [-4., -4., -5., -5., -1., -1., -0.1, -0.1], deepcopy(nonparametric), activation_function, fill(true, 8), Float64[], fill(true, 8)
        @success out == (nothing, [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])

        @params [-1., 2., -3., 4., 4., 4., 5., 5., 1., 1., 0.1, 0.1], deepcopy(semiparametric), activation_function, vcat([true, false, true, false], fill(false, 8)), Float64[], fill(true, 12)
        @success out == ([1., 2., 3., 4.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])

        @params [2., 4., 0.1, 0.1], deepcopy(lin_model), activation_function, fill(false, 6), [1., 3.], vcat([false, true, false, true], fill(true, 2))
        @params [2., 4., 0.1, 0.1], deepcopy(nonlin_model), activation_function, fill(false, 6), [1., 3.], vcat([false, true, false, true], fill(true, 2))
        @success out == ([1., 2., 3., 4.], nothing, nothing, [0.1, 0.1])

        @params [5., 5., 1., 0.1, 0.1], deepcopy(nonparametric), activation_function, fill(false, 8), [4., 4., 1.], vcat([false, false, true, true], [false, true], fill(true, 2))
        @success out == (nothing, [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])

        @params [2., 4., 4., 4., 5., 5., 1., 1., 0.1, 0.1], deepcopy(semiparametric), activation_function, fill(false, 12), [1., 3.], vcat([false, true, false, true], fill(true, 8))
        @success out == ([1., 2., 3., 4.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])

        @params [3., -4., 0.1, 0.1], deepcopy(lin_model), activation_function, vcat([false, true, false, true], fill(false, 2)), [1., 2.], vcat([false, false, true, true], fill(true, 2))
        @params [3., -4., 0.1, 0.1], deepcopy(nonlin_model), activation_function, vcat([false, true, false, true], fill(false, 2)), [1., 2.], vcat([false, false, true, true], fill(true, 2))
        @success out == ([1., 2., 3., 4.], nothing, nothing, [0.1, 0.1])

        @params [-5., -5., -1., -0.1, -0.1], deepcopy(nonparametric), activation_function, fill(true, 8), [4., 4., 1.], vcat([false, false, true, true], [false, true], fill(true, 2))
        @success out == (nothing, [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])

        @params [3., -4., 4., 4., 5., 5., 1., 1., 0.1, 0.1], deepcopy(semiparametric), activation_function, vcat([false, true, false, true], fill(false, 8)), [1., 2.], vcat([false, false, true, true], fill(true, 8))
        @success out == ([1., 2., 3., 4.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])
    end
end

@testset "devectorize_params(params, model)" begin
    lin_model = LinModel(;
        lift = (x) -> [
            [sin(x[1]), exp(x[2])],
            [cos(x[1]), exp(x[2])],
        ],
        theta_priors = fill(Normal(), 4),
        noise_std_priors = fill(Dirac(0.1), 2),
    )
    nonlin_model = NonlinModel(;
        predict = (x, θ) -> [
            θ[1] * sin(x[1]) + θ[2] * exp(x[2]),
            θ[3] * cos(x[1]) + θ[4] * exp(x[2]),
        ],
        theta_priors = fill(Normal(), 4),
        noise_std_priors = fill(Dirac(0.1), 2),
    )
    nonparametric = Nonparametric(;
        kernel = Matern32Kernel(),
        amp_priors = fill(LogNormal(), 2),
        length_scale_priors = fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2),
        noise_std_priors = fill(Dirac(0.1), 2),
    )
    semiparametric = Semiparametric(;
        parametric = nonlin_model,
        nonparametric = nonparametric,
    )

    @param_test BOSS.devectorize_params begin
        @params [1., 2., 3., 4., 0.1, 0.1], deepcopy(lin_model)
        @params [1., 2., 3., 4., 0.1, 0.1], deepcopy(nonlin_model)
        @success out == ([1., 2., 3., 4.], nothing, nothing, [0.1, 0.1])

        @params [4., 4., 5., 5., 1., 1., 0.1, 0.1], deepcopy(nonparametric)
        @success out == (nothing, [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])

        @params [1., 2., 3., 4., 4., 4., 5., 5., 1., 1., 0.1, 0.1], deepcopy(semiparametric)
        @success out == ([1., 2., 3., 4.], [4.;4.;; 5.;5.;;], [1., 1.], [0.1, 0.1])
    end
end

@testset "create_activation_mask(param_counts, mask_hyperparams, mask_params)" begin
    @param_test BOSS.create_activation_mask begin
        @params (3,6,2,2), false, false
        @params (3,6,2,2), fill(false, 3), false
        @success out == fill(false, 13)

        @params (3,6,2,2), false, true
        @params (3,6,2,2), fill(false, 3), true
        @success out == vcat(fill(false, 3), fill(true, 10))

        @params (3,6,2,2), true, false
        @params (3,6,2,2), fill(true, 3), false
        @success out == vcat(fill(true, 3), fill(false, 10))

        @params (3,6,2,2), [true, false, true], false
        @success out == vcat([true, false, true], fill(false, 10))

        @params (0,6,2,2), false, false
        @params (0,6,2,2), true, false
        @params (0,6,2,2), Bool[], false
        @success out == fill(false, 10)

        @params (0,6,2,2), false, true
        @params (0,6,2,2), true, true
        @params (0,6,2,2), Bool[], true
        @success out == fill(true, 10)

        @params (3,6,2,2), false, false
        @params (3,6,2,2), fill(false, 3), false
        @success out == fill(false, 13)

        @params (3,6,2,2), false, true
        @params (3,6,2,2), fill(false, 3), true
        @success out == vcat(fill(false, 3), fill(true, 10))

        @params (3,6,2,2), [true, false, true], false
        @success out == vcat([true, false, true], fill(false, 10))

        @params (3,6,2,2), [true, false, true], true
        @success out == vcat([true, false, true], fill(true, 10))
    end
end

@testset "create_dirac_skip_mask(priors)" begin
    @param_test BOSS.create_dirac_skip_mask begin
        @params (fill(Normal(), 4), nothing, nothing, fill(LogNormal(), 2)) |> Tuple
        @success out == (fill(true, 6), Float64[])

        @params ([Dirac(1.), Normal(), Dirac(3.), Normal()], nothing, nothing, fill(LogNormal(), 2)) |> Tuple
        @success out == (vcat([false, true, false, true], fill(true, 2)), [1., 3.])

        @params (fill(Normal(), 4), MultivariateDistribution[], UnivariateDistribution[], fill(Dirac(0.1), 2)) |> Tuple
        @success out == (vcat(fill(true, 4), [false, false]), [0.1, 0.1])

        @params (UnivariateDistribution[], fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2), fill(LogNormal(), 2), fill(LogNormal(), 2)) |> Tuple
        @success out == (fill(true, 8), Float64[])

        @params (UnivariateDistribution[], [product_distribution(fill(Dirac(1.), 2)), BOSS.mvlognormal([1., 1.], [1., 1.])], [Dirac(1.), LogNormal()], fill(LogNormal(), 2)) |> Tuple
        @success out == (vcat([false, false, true, true], [false, true], fill(true,  2)), [1., 1., 1.])

        @params (UnivariateDistribution[], fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2), fill(LogNormal(), 2), fill(Dirac(0.1), 2)) |> Tuple
        @success out == (vcat(fill(true, 6), fill(false, 2)), [0.1, 0.1])

        @params (fill(Normal(), 4), fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2), fill(LogNormal(), 2), fill(LogNormal(), 2)) |> Tuple
        @success out == (fill(true, 12), Float64[])

        @params ([Dirac(1.), Normal(), Dirac(3.), Normal()], fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2), fill(LogNormal(), 2), fill(LogNormal(), 2)) |> Tuple
        @success out == (vcat([false, true, false, true], fill(true, 8)), [1., 3.])

        @params (fill(Normal(), 4), [product_distribution(fill(Dirac(1.), 2)), BOSS.mvlognormal([1., 1.], [1., 1.])], [Dirac(1.), LogNormal()], fill(LogNormal(), 2)) |> Tuple
        @success out == (vcat(fill(true, 4), [false, false, true, true], [false, true], fill(true, 2)), [1., 1., 1.])

        @params (fill(Normal(), 4), fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2), fill(LogNormal(), 2), fill(Dirac(0.1), 2)) |> Tuple
        @success out == (vcat(fill(true, 10), [false, false]), [0.1, 0.1])

        @params ([Dirac(1.), Normal(), Dirac(3.), Normal()], fill(BOSS.mvlognormal([1., 1.], [1., 1.]), 2), fill(LogNormal(), 2), fill(Dirac(0.1), 2)) |> Tuple
        @success out == (vcat([false, true, false, true], fill(true, 4), fill(true, 2), [false, false]), [1., 3., 0.1, 0.1])
    end
end
