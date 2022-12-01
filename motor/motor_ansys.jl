using Random
using Combinatorics
using Distributions
using KernelFunctions
using Optim
include("../src/boss.jl")

# Parametric Surrogate Model
# calc: [nk, dk, Ds] -> [Dp, T_av, S_stator]
include("./param_model.jl")

# cols: nk, dk, Ds, Dp, T_av
function init_data()
    data = [
        34.0	25.0	474.2	768.1	123.0
        42.0	17.0	481.5	2782.3	104.5
        40.0	11.7	433.8	15541.5	84.0
        30.0	19.0	485.2	3507.1	114.1
        46.0	16.3	444.8	2731.0	94.3
        32.0	23.7	437.5	1119.7	113.5
        42.0	10.3	477.8	24373.9	87.3
        50.0	13.0	415.5	6245.2	82.2
        58.0	15.0	441.2	2498.1	88.2
        54.0	19.7	470.5	854.5	105.4
        44.0	23.0	455.8	656.6	110.9
        38.0	27.7	430.2	375.1	115.9
        32.0	15.7	452.2	6945.7	98.8
        34.0	12.3	496.2	17703.5	99.0
        40.0	29.7	466.8	235.6	131.3
        36.0	17.7	419.2	3220.0	97.0
        52.0	11.0	463.2	11869.6	84.0
        44.0	21.7	411.8	858.7	99.3
        56.0	14.3	492.5	3275.6	97.9
    ]
    data[:,2:3] /= 1000  # [mm] to [m]
    return data
end

function def_model()
    #                 Q, D1, D2, l, t, alt, alfa_0, lam_fe, Pl, alfa_a, alfa_b
    def_param_vals = [0.5, 0.297, 0.4, 0.23, 30, 325, 16, 29, 5000, 1., 0.]

    # # 8-arg version
    # predict = (nk, dk, Ds, Q, D1, D2, l, t, alt, alfa_0, lam_fe) -> MotorParam.calc(nk, dk, Ds; Q, D1, D2, l, t, alt, alfa_0, lam_fe)
    # param_priors = vcat(Beta(2,2), construct_lognormal.(def_param_vals[2:8], def_param_vals[2:8] ./ 2))
    # param_count = 8

    # Pl, alfa_duct version
    predict = (nk, dk, Ds, Pl, alfa_a, alfa_b) -> MotorParam.calc(nk, dk, Ds; Pl, alfa_a, alfa_b)
    param_priors = [truncated(Normal(10_000, 5_000); lower=0.), LogNormal(0., 0.5), Normal(0., 50.)]
    param_count = 3

    param_model = Boss.NonlinModel(
        (x, params) -> predict(x..., params...)[1:2],
        param_priors,
        param_count,
    )
end

function fit_model(data) #param_idx=collect(1:8)
    # DATA - - - - - -
    init_X = data[:,1:3]' |> collect
    init_Y = data[:,4:5]' |> collect
    x_dim = size(init_X)[1]
    y_dim = size(init_Y)[1]

    # HYPERPARAMS - - - - - -
    param_fit_alg = :BI
    mc_settings = Boss.MCSettings(400, 10, 8, 5)

    kernel = Boss.DiscreteKernel(Matern52Kernel(), [true, false, false])
    gp_params_priors = [MvLogNormal(ones(x_dim), ones(x_dim)) for _ in 1:y_dim]
    noise_priors = [LogNormal(-2.3, 1.) for _ in 1:y_dim]

    # MODEL - - - - - -
    param_model = def_model()

    # FIT MODEL - - - - - -
    # fit, _, params, noise = Boss.fit_parametric_model(init_X, init_Y, param_model, noise_priors; param_fit_alg, mc_settings, parallel=true, info=true, debug=false)
    fit, _, params, _, noise = Boss.fit_semiparametric_model(init_X, init_Y, param_model, kernel, gp_params_priors, noise_priors; param_fit_alg, mc_settings, parallel=true, info=true, debug=false)

    return fit, params, noise
end

function crosscheck()
    Random.seed!(5555)

    tst_size = 4
    total_size = 19
    n_sets = 8

    tst_sets = rand(combinations(collect(1:total_size), tst_size) |> collect, n_sets)

    errs = []    
    for tst_idx in tst_sets
        println()
        @show tst_idx
        trn_idx = [i for i in 1:total_size if !in(i, tst_idx)]

        data = init_data()
        train = data[trn_idx,:]
        test = data[tst_idx,:]

        model, _, _ = fit_model(train)

        preds = vcat((x -> model(x)[1]').(eachrow(test[:,1:3]))...)
        err = rms_err.(eachcol(test[:,4:5]), eachcol(preds))
        @show err
        push!(errs, err)
    end

    println()
    return mean(errs)
end

rms_err(tru::AbstractVector, pred::AbstractVector) = sqrt(sum((tru .- pred) .^ 2) / length(tru))

function fit_error(fit, data)
    preds = vcat((x -> fit(x)[1]').(eachrow(data[:,1:3]))...)
    rms_err.(eachcol(data[:,4:5]), eachcol(preds))
end

function construct_lognormal(mean, var)
    # mean = exp(μ + (σ2 / 2))
    # var = (exp(σ2) - 1) * exp(2*μ + σ2)
    σ = sqrt(log(var / mean ^ 2 + 1))
    μ = log(mean) - σ ^ 2 / 2
    LogNormal(μ, σ)
end
