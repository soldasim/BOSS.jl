using Random
using Combinatorics
using Distributions
using KernelFunctions
using Optim
using NLopt
using Evolutionary

include("../src/boss.jl")
include("./param_model_eliptic.jl")

# cols: nk, dk, Ds, Dp, T_av
function get_data()
    data = [
        # initial data
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
domain_convert() = [1.,1000.,1000.]

function split_data(data)
    init_X = data[:,1:3]' |> collect
    init_Y = data[:,4:5]' |> collect
    return init_X, init_Y
end

function def_model()
    predict = (nk, dk, Ds, Pl, alfa_a, alfa_b) -> MotorParam.calc(nk, dk, Ds; Pl, alfa_a, alfa_b)
    param_priors = [truncated(Normal(10_000, 5_000); lower=0.), LogNormal(0., 0.5), Normal(0., 50.)]
    param_count = 3

    param_model = Boss.NonlinModel(
        (x, params) -> predict(x..., params...)[1:2],
        param_priors,
        param_count,
    )
end

function opt_acq(data; model_samples=nothing)
    # DATA
    init_X, init_Y = split_data(data)
    x_dim = size(init_X)[1]
    y_dim = size(init_Y)[1]

    # HYPERPARAMS
    x_tol = 0.05

    # PROBLEM DEF
    domain = MotorParam.domain()
    discrete_dims = [true, false, false]
    constraints = [1000., Inf]
    fitness = Boss.LinFitness([0., -1.])

    # FIT MODEL
    if isnothing(model_samples)
        _, model_samples, _, _ = fit_model(data; discrete_dims)
    end

    # OPTIMIZE ACQ
    bsf = Boss.get_best_yet(fitness, init_X, init_Y, domain, constraints)
    ϵ_samples = rand(Normal(), (y_dim, length(model_samples)))
    acq = Boss.construct_acq_from_samples(fitness, model_samples, constraints, ϵ_samples, bsf)

    domain_opt = domain[1] .* domain_convert(), domain[2] .* domain_convert()
    function acq_opt(x)
        val = acq(x ./ domain_convert())
        isnan(val) ? 0. : val  # TODO
    end

    # optimize acq with COBYLA
    nlopt = Opt(:LN_COBYLA, x_dim)
    nlopt.xtol_abs = x_tol
    multistart = 200
    nlopt.maxeval = 1_000

    function nlopt_c!(result::Vector, x::Vector, grad::Matrix)
        x ./= domain_convert()
        if length(grad) > 0
            grad .= ForwardDiff.jacobian(MotorParam.domain_constraints, x)'
        end
        result .= MotorParam.domain_constraints(x)
    end
    inequality_constraint!(nlopt, nlopt_c!, zeros(3))
    
    Boss.opt_acq_NLopt(acq_opt, domain_opt; x_dim, multistart, discrete_dims, optimizer=nlopt, parallel=false, info=true)
end

function fit_model(data; discrete_dims=[true, false, false])
    # DATA
    init_X, init_Y = split_data(data)
    x_dim = size(init_X)[1]
    y_dim = size(init_Y)[1]

    # HYPERPARAMS
    mc_settings = Boss.MCSettings(PG(20), 400, 10, 8, 5)
    kernel = Boss.DiscreteKernel(Matern52Kernel(), discrete_dims)
    gp_params_priors = [MvLogNormal(ones(x_dim), ones(x_dim)) for _ in 1:y_dim]
    noise_priors = [LogNormal(-2.3, 1.) for _ in 1:y_dim]
    # noise_priors = [LogNormal(-10., 1.) for _ in 1:y_dim]  # no noise

    # FIT MODEL
    param_model = def_model()
    # fit, model_samples, params, noise = Boss.fit_parametric_model(init_X, init_Y, param_model, noise_priors; param_fit_alg, mc_settings, parallel=true, info=true, debug=false)
    fit, model_samples, params, gp_params, noise = Boss.fit_semiparametric_model(init_X, init_Y, param_model, kernel, gp_params_priors, noise_priors; param_fit_alg=:BI, mc_settings, parallel=true, info=true, debug=false)

    return fit, model_samples, params, noise
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

        data = get_data()
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
