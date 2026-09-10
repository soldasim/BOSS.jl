
# ---------------------------------------------------------------------------
# `TransformedModel`'s `predictive_kind`/`predictive_samples` across the base model's
# `predictive_kind` (Gaussian/Sampled) × `output_transform` (nothing/Sliced/Joint) combinations,
# and (separately) `DefaultModelPosterior`/`DefaultModelPosteriorSlice`'s own `predictive_samples`
# redirects.
# ---------------------------------------------------------------------------

# A `WarpedGaussianProcess` with an identity warp -- a real, slice-native `SampledPredictive` model.
function _sampled_slice_native_setup()
    model = WarpedGaussianProcess(;
        amplitude_priors = fill(LogNormal(), 1),
        lengthscale_priors = fill(BOSS.mvlognormal([1.], [1.]), 1),
        noise_std_priors = fill(Dirac(1e-4), 1),
        output_warpings = [AffineWarping(; shift_prior=Dirac(0.), scale_prior=Dirac(1.))],
    )
    params = WarpedGaussianProcessParams(reshape([1.], 1, 1), [1.], [1e-4], [[0., 1.]])
    X = reshape(collect(1.:5.), 1, :)
    Y = reshape(sin.(X[1,:]) .+ 2., 1, :)
    data = ExperimentData(X, Y)
    return model, params, data
end

# A `GaussianProcess` -- a real, slice-native `GaussianPredictive` model.
function _gaussian_slice_native_setup()
    model = GaussianProcess(;
        amplitude_priors = fill(LogNormal(), 1),
        lengthscale_priors = fill(BOSS.mvlognormal([1.], [1.]), 1),
        noise_std_priors = fill(Dirac(1e-4), 1),
    )
    params = GaussianProcessParams(reshape([1.], 1, 1), [1.], [1e-4])
    X = reshape(collect(1.:5.), 1, :)
    Y = reshape(sin.(X[1,:]) .+ 2., 1, :)
    data = ExperimentData(X, Y)
    return model, params, data
end

# `forward` must implement both a pointwise `(y_) -> y` method and a mean/std propagation
# `(y_, std_) -> (y, std)` method (see `OutputTransform`'s docstring).
function _exp_sliced_transform()
    f(y_::Real) = exp(y_)
    f(y_::Real, s_::Real) = (exp(y_), s_ * exp(y_))
    b(y::Real) = log(y)
    return SlicedOutputTransform([f], [b])
end
function _exp_joint_transform()
    f(y_::AbstractVector) = exp.(y_)
    f(y_::AbstractVector, s_::AbstractVector) = (exp.(y_), s_ .* exp.(y_))
    b(y::AbstractVector) = log.(y)
    return JointOutputTransform(f, b)
end

@testset "TransformedModel: predictive_kind/predictive_samples, real slice-native base models" begin
    x = [3.]
    Xtest = reshape([2., 3., 4.], 1, :)

    for (label, out_kw) in (
        ("Nothing", NamedTuple()),
        ("Sliced", (; output_transform=_exp_sliced_transform())),
        ("Joint", (; output_transform=_exp_joint_transform())),
    )
        @testset "OUT=$label" begin
            # --- Sampled base (WarpedGaussianProcess), slice-native ---
            base, base_params, data = _sampled_slice_native_setup()
            tm = TransformedModel(; base_model=base, out_kw...)
            @test predictive_kind(typeof(tm)) isa SampledPredictive

            tparams = TransformedParams(base_params)
            tpost_slice = model_posterior_slice(tm, tparams, data, 1)
            @test predictive_kind(tpost_slice) isa SampledPredictive

            if label == "Joint"
                # `JointOutputTransform` is never sliceable (see `sliceable`), so "slice access"
                # here isn't a real per-dimension posterior -- it's the generic
                # `DefaultModelPosteriorSlice` fallback wrapping the *joint* `TransformedPosterior`,
                # which has exactly the same limitation as the joint level itself (below): the
                # base only samples per-dimension, so there is no joint sample to wrap at all.
                @test_throws MethodError predictive_samples(tpost_slice, x)
            else
                ys, ws = predictive_samples(tpost_slice, x)
                @test ws isa AbstractVector{<:Real}
                @test isapprox(sum(ws), 1.; atol=1e-8)

                Ys, Ws = predictive_samples(tpost_slice, Xtest)
                @test size(Ys) == size(Ws)
                @test all(isapprox.(vec(sum(Ws; dims=1)), 1.; atol=1e-8))

                if label == "Nothing"
                    # no transform in the way => the weighted mean must exactly match `mean_and_var`
                    m, _ = mean_and_var(tpost_slice, x)
                    @test isapprox(sum(ws .* ys), m; atol=1e-6)
                    mX, _ = mean_and_var(tpost_slice, Xtest)
                    @test isapprox(vec(sum(Ws .* Ys; dims=1)), mX; atol=1e-6)
                end
                # (for Sliced, `mean_and_var` uses a *different*, delta-method-approximated path
                # -- see `OutputTransform`'s docstring -- so it is not expected to match
                # `predictive_samples`'s exact per-atom transform; only `Nothing` is an exact check.)
            end

            # base only samples at the slice level => joint-level predictive_samples must fail,
            # regardless of `output_transform` (this is the pre-existing, accepted contract of
            # `predictive_kind`: it says "available somewhere", not "at every level").
            tpost_joint = model_posterior(tm, tparams, data)
            @test predictive_kind(tpost_joint) isa SampledPredictive
            @test_throws MethodError predictive_samples(tpost_joint, x)

            # --- Gaussian base (GaussianProcess), slice-native ---
            gbase, gbase_params, gdata = _gaussian_slice_native_setup()
            gtm = TransformedModel(; base_model=gbase, out_kw...)
            @test predictive_kind(typeof(gtm)) isa GaussianPredictive

            gtparams = TransformedParams(gbase_params)
            gpost_slice = model_posterior_slice(gtm, gtparams, gdata, 1)
            @test predictive_kind(gpost_slice) isa GaussianPredictive
            @test_throws MethodError predictive_samples(gpost_slice, x)

            m, v = mean_and_var(gpost_slice, x)
            @test m isa Real
            @test v isa Real
        end
    end
end


# ---------------------------------------------------------------------------
# Minimal, test-only *joint-native* models (implementing `model_posterior` directly rather than
# `model_posterior_slice`). No model in the package itself is joint-native -- every real model
# models each output dimension independently -- so these stubs are the only way to exercise the
# joint-level code paths: `DefaultModelPosteriorSlice`'s redirect, and `TransformedModel`'s
# joint-level `predictive_samples` for a genuinely joint base.
# ---------------------------------------------------------------------------

struct _ToyJointGaussianModel <: SurrogateModel end
struct _ToyJointGaussianParams <: ModelParams{_ToyJointGaussianModel} end
struct _ToyJointGaussianPosterior <: ModelPosterior{_ToyJointGaussianModel} end

BOSS.model_posterior(::_ToyJointGaussianModel, ::_ToyJointGaussianParams, ::ExperimentData) =
    _ToyJointGaussianPosterior()

BOSS.mean(::_ToyJointGaussianPosterior, x::AbstractVector{<:Real}) = [sum(x), 2*sum(x)]
BOSS.mean(post::_ToyJointGaussianPosterior, X::AbstractMatrix{<:Real}) =
    hcat([mean(post, x) for x in eachcol(X)]...)
BOSS.var(::_ToyJointGaussianPosterior, x::AbstractVector{<:Real}) = [1., 1.]
BOSS.var(post::_ToyJointGaussianPosterior, X::AbstractMatrix{<:Real}) =
    hcat([var(post, x) for x in eachcol(X)]...)
_toy_gaussian_cov() = [1. 0.5; 0.5 1.]
BOSS.cov(::_ToyJointGaussianPosterior, X::AbstractMatrix{<:Real}) =
    cat([_toy_gaussian_cov() for _ in axes(X, 2)]...; dims=3)
BOSS.mean_and_var(post::_ToyJointGaussianPosterior, x::AbstractVector{<:Real}) = (mean(post, x), var(post, x))
BOSS.mean_and_var(post::_ToyJointGaussianPosterior, X::AbstractMatrix{<:Real}) = (mean(post, X), var(post, X))
BOSS.mean_and_cov(post::_ToyJointGaussianPosterior, X::AbstractMatrix{<:Real}) = (mean(post, X), cov(post, X))

struct _ToyJointSampledModel <: SurrogateModel end
struct _ToyJointSampledParams <: ModelParams{_ToyJointSampledModel} end
struct _ToyJointSampledPosterior <: ModelPosterior{_ToyJointSampledModel} end

BOSS.predictive_kind(::Type{<:_ToyJointSampledModel}) = SampledPredictive()
BOSS.model_posterior(::_ToyJointSampledModel, ::_ToyJointSampledParams, ::ExperimentData) =
    _ToyJointSampledPosterior()

# 4 fixed joint atoms with a genuine cross-dimension dependency (dim 2 = 2 * dim 1 exactly),
# unlike bundled-independent per-dimension slices. Shared weights (sum to 1). Ignores `x`/data
# for simplicity -- only the API contract is under test here, not a real model fit.
const _TOY_YS = [1. 2. 3. 4.; 2. 4. 6. 8.]
const _TOY_WS = reshape([.1, .4, .4, .1], 1, :)

BOSS.predictive_samples(::_ToyJointSampledPosterior, ::AbstractVector{<:Real}; kwargs...) = (_TOY_YS, _TOY_WS)
function BOSS.predictive_samples(::_ToyJointSampledPosterior, X::AbstractMatrix{<:Real}; kwargs...)
    n = size(X, 2)
    return cat(fill(_TOY_YS, n)...; dims=3), cat(fill(_TOY_WS, n)...; dims=3)
end

BOSS.mean(::_ToyJointSampledPosterior, x::AbstractVector{<:Real}) = vec(sum(_TOY_WS .* _TOY_YS; dims=2))
BOSS.mean(post::_ToyJointSampledPosterior, X::AbstractMatrix{<:Real}) =
    hcat([mean(post, x) for x in eachcol(X)]...)
function BOSS.var(post::_ToyJointSampledPosterior, x::AbstractVector{<:Real})
    m = mean(post, x)
    return vec(sum(_TOY_WS .* _TOY_YS .^ 2; dims=2)) .- m .^ 2
end
BOSS.var(post::_ToyJointSampledPosterior, X::AbstractMatrix{<:Real}) =
    hcat([var(post, x) for x in eachcol(X)]...)
function BOSS.cov(post::_ToyJointSampledPosterior, X::AbstractMatrix{<:Real})
    m = mean(post, X[:, 1])
    K = size(_TOY_YS, 2)
    c = zeros(2, 2)
    for k in 1:K
        d = _TOY_YS[:, k] .- m
        c .+= _TOY_WS[1, k] .* (d * d')
    end
    return cat([c for _ in axes(X, 2)]...; dims=3)
end
BOSS.mean_and_var(post::_ToyJointSampledPosterior, x::AbstractVector{<:Real}) = (mean(post, x), var(post, x))
BOSS.mean_and_var(post::_ToyJointSampledPosterior, X::AbstractMatrix{<:Real}) = (mean(post, X), var(post, X))
BOSS.mean_and_cov(post::_ToyJointSampledPosterior, X::AbstractMatrix{<:Real}) = (mean(post, X), cov(post, X))

_toy_data() = ExperimentData(reshape(collect(1.:5.), 1, :), vcat((1.:5.)', (2.:2.:10.)'))

@testset "DefaultModelPosteriorSlice: predictive_samples redirect (safe direction)" begin
    data = _toy_data()
    post_slice2 = model_posterior_slice(_ToyJointSampledModel(), _ToyJointSampledParams(), data, 2)
    @test post_slice2 isa BOSS.DefaultModelPosteriorSlice
    @test predictive_kind(post_slice2) isa SampledPredictive

    x = [3.]
    ys, ws = predictive_samples(post_slice2, x)
    @test ys == _TOY_YS[2, :]  # dim 2's true atom values, not re-derived from dim 1
    @test ws == vec(_TOY_WS)
    m, _ = mean_and_var(post_slice2, x)
    @test isapprox(sum(ws .* ys), m; atol=1e-8)
end

@testset "DefaultModelPosterior: no predictive_samples redirect (unsafe direction)" begin
    # A slice-native `SampledPredictive` model's *joint* posterior must not synthesize samples
    # by bundling its independent per-dimension slices.
    model, params, data = _sampled_slice_native_setup()
    post = model_posterior(model, params, data)
    @test post isa BOSS.DefaultModelPosterior
    @test_throws MethodError predictive_samples(post, [3.])
end

function _toy_sliced_transform()
    f(y_::Real) = exp(y_)
    f(y_::Real, s_::Real) = (exp(y_), s_ * exp(y_))
    b(y::Real) = log(y)
    return SlicedOutputTransform([f, f], [b, b])
end
function _toy_joint_transform()
    f(y_::AbstractVector) = exp.(y_)
    f(y_::AbstractVector, s_::AbstractVector) = (exp.(y_), s_ .* exp.(y_))
    b(y::AbstractVector) = log.(y)
    return JointOutputTransform(f, b)
end

@testset "TransformedModel: predictive_kind/predictive_samples, toy joint-native base models" begin
    data = _toy_data()
    x = [3.]

    for (label, out_kw) in (
        ("Nothing", NamedTuple()),
        ("Sliced", (; output_transform=_toy_sliced_transform())),
        ("Joint", (; output_transform=_toy_joint_transform())),
    )
        @testset "OUT=$label" begin
            # --- Sampled joint-native base ---
            tm = TransformedModel(; base_model=_ToyJointSampledModel(), out_kw...)
            @test predictive_kind(typeof(tm)) isa SampledPredictive
            tparams = TransformedParams(_ToyJointSampledParams())

            tpost_joint = model_posterior(tm, tparams, data)
            Ys, Ws = predictive_samples(tpost_joint, x)
            @test size(Ys, 1) == 2  # y_dim
            @test isapprox(sum(Ws; dims=2)[1], 1.; atol=1e-8)

            tpost_slice = model_posterior_slice(tm, tparams, data, 2)
            ys, ws = predictive_samples(tpost_slice, x)
            # joint-level and slice-level access must agree on dimension 2, regardless of transform
            @test isapprox(ys, Ys[2, :]; atol=1e-8)
            @test isapprox(ws, vec(Ws[1, :]); atol=1e-8)

            if label == "Nothing"
                m, _ = mean_and_var(tpost_joint, x)
                @test isapprox(vec(sum(Ws .* Ys; dims=2)), m; atol=1e-6)
            end

            # --- Gaussian joint-native base ---
            gtm = TransformedModel(; base_model=_ToyJointGaussianModel(), out_kw...)
            @test predictive_kind(typeof(gtm)) isa GaussianPredictive
            gtparams = TransformedParams(_ToyJointGaussianParams())

            gpost_joint = model_posterior(gtm, gtparams, data)
            m, v = mean_and_var(gpost_joint, x)
            @test m isa AbstractVector{<:Real}
            @test_throws MethodError predictive_samples(gpost_joint, x)

            gpost_slice = model_posterior_slice(gtm, gtparams, data, 1)
            m1, v1 = mean_and_var(gpost_slice, x)
            @test m1 isa Real
        end
    end
end
