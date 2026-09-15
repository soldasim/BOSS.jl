
# Representative warpings paired with valid parameter vectors `θ`.
# (Includes Yeo-Johnson at λ ∈ {0, 2} to exercise the logarithmic limit branches.)
const WARP_CASES = [
    (AffineWarping(; shift_prior=Normal(), scale_prior=LogNormal()), [0.7, 2.0]),
    (YeoJohnsonWarping(; λ_prior=truncated(Normal(1, 0.5); lower=0, upper=2)), [0.5]),
    (YeoJohnsonWarping(; λ_prior=truncated(Normal(1, 0.5); lower=0, upper=2)), [0.0]),  # log limit (y ≥ 0)
    (YeoJohnsonWarping(; λ_prior=truncated(Normal(1, 0.5); lower=0, upper=2)), [2.0]),  # log limit (y < 0)
    (SinhArcsinhWarping(; skewness_prior=Normal(), tailweight_prior=LogNormal()), [0.3, 1.4]),
    (ComposedWarping(
        YeoJohnsonWarping(; λ_prior=truncated(Normal(1, 0.5); lower=0, upper=2)),
        AffineWarping(; shift_prior=Normal(), scale_prior=LogNormal()),
    ), [0.6, 0.2, 1.5]),
]

const WARP_YS = [-2.3, -0.7, 0.0, 0.4, 3.1]

@testset "warp_inverse ∘ warp_forward ≈ identity" begin
    for (w, θ) in WARP_CASES, y in WARP_YS
        δ = warp_forward(w, θ, y)
        @test isapprox(warp_inverse(w, θ, δ), y; atol=1e-6)
    end
end

@testset "warp_logderiv matches autodiff of warp_forward" begin
    for (w, θ) in WARP_CASES, y in WARP_YS
        ad = log(abs(BOSS.ForwardDiff.derivative(yy -> warp_forward(w, θ, yy), y)))
        @test isapprox(warp_logderiv(w, θ, y), ad; atol=1e-6)
    end
end

@testset "forward warp is strictly monotone increasing" begin
    for (w, θ) in WARP_CASES
        δs = [warp_forward(w, θ, y) for y in -3.0:0.2:3.0]
        @test all(diff(δs) .> 0)
    end
end

@testset "identity parameters give the identity warp" begin
    @param_test (w, θ, y) -> warp_forward(w, θ, y) begin
        # Yeo-Johnson: λ = 1
        @params YeoJohnsonWarping(; λ_prior=Dirac(1.)), [1.0], -2.3
        @params YeoJohnsonWarping(; λ_prior=Dirac(1.)), [1.0], 0.0
        @params YeoJohnsonWarping(; λ_prior=Dirac(1.)), [1.0], 3.1
        # Sinh-arcsinh: a = 0, b = 1
        @params SinhArcsinhWarping(; skewness_prior=Dirac(0.), tailweight_prior=Dirac(1.)), [0.0, 1.0], -2.3
        @params SinhArcsinhWarping(; skewness_prior=Dirac(0.), tailweight_prior=Dirac(1.)), [0.0, 1.0], 3.1
        # Affine: shift = 0, scale = 1
        @params AffineWarping(; shift_prior=Dirac(0.), scale_prior=Dirac(1.)), [0.0, 1.0], -2.3
        @params AffineWarping(; shift_prior=Dirac(0.), scale_prior=Dirac(1.)), [0.0, 1.0], 3.1
        @success isapprox(out, in[3]; atol=1e-8)
    end
end

@testset "warp_param_count and warp_param_priors" begin
    affine = AffineWarping(; shift_prior=Normal(), scale_prior=LogNormal())
    yj = YeoJohnsonWarping(; λ_prior=truncated(Normal(1, 0.5); lower=0, upper=2))
    sa = SinhArcsinhWarping(; skewness_prior=Normal(), tailweight_prior=LogNormal())
    composed = ComposedWarping(yj, sa, affine)

    @test warp_param_count(affine) == 2
    @test warp_param_count(yj) == 1
    @test warp_param_count(sa) == 2
    @test warp_param_count(composed) == 5
    @test length(warp_param_priors(composed)) == 5
    # Composed priors are the concatenation of the sub-warpings' priors, in order.
    @test warp_param_priors(composed) == vcat(
        warp_param_priors(yj), warp_param_priors(sa), warp_param_priors(affine),
    )
end

@testset "ComposedWarping applies sub-warpings in forward order" begin
    yj = YeoJohnsonWarping(; λ_prior=Dirac(0.5))
    affine = AffineWarping(; shift_prior=Dirac(1.), scale_prior=Dirac(2.))
    composed = ComposedWarping(yj, affine)

    @param_test (y,) -> warp_forward(composed, [0.5, 1.0, 2.0], y) begin
        @params 0.4
        @params 1.7
        @params -0.6
        # forward = affine(yj(y)): apply yj first, then affine.
        @success isapprox(out, warp_forward(affine, [1.0, 2.0], warp_forward(yj, [0.5], in[1])); atol=1e-10)
    end
end

@testset "default priors" begin
    # All warps construct with no arguments (identity-centered defaults).
    @test AffineWarping() isa AffineWarping
    @test SinhArcsinhWarping() isa SinhArcsinhWarping
    @test YeoJohnsonWarping() isa YeoJohnsonWarping

    # Defaults are centered at the identity transform (the positive-scale priors are
    # truncated only from below, so their median sits slightly above 1).
    @test mean(AffineWarping().shift_prior) == 0.0         # shift = 0
    @test mean(SinhArcsinhWarping().skewness_prior) == 0.0 # a = 0
    @test isapprox(median(SinhArcsinhWarping().tailweight_prior), 1.0; atol=0.05)  # b ≈ 1

    # The YJ default prior is restricted to the finite-moment range [0, 2], centered at λ = 1.
    yj = YeoJohnsonWarping()
    @test minimum(yj.λ_prior) == 0.0
    @test maximum(yj.λ_prior) == 2.0
    @test isapprox(median(yj.λ_prior), 1.0; atol=1e-6)
end

@testset "YeoJohnsonWarping warns iff λ_prior support exceeds [0, 2]" begin
    # In-range priors: no warning.
    @test_logs YeoJohnsonWarping()                                                  # default
    @test_logs YeoJohnsonWarping(; λ_prior=Dirac(1.0))
    @test_logs YeoJohnsonWarping(; λ_prior=truncated(Normal(1, 0.5); lower=0, upper=2))

    # Support extending beyond [0, 2]: warning.
    @test_logs (:warn,) YeoJohnsonWarping(; λ_prior=Normal(1, 0.5))                 # unbounded
    @test_logs (:warn,) YeoJohnsonWarping(; λ_prior=truncated(Normal(1, 0.5); lower=-1, upper=2))
    @test_logs (:warn,) YeoJohnsonWarping(; λ_prior=truncated(Normal(1, 0.5); lower=0, upper=3))
end
