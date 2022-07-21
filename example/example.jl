using Random
using Distributions

include("../boss.jl")
include("data.jl")

# Load model
include("models/model-expcos.jl")
include("models/model-lincos.jl")
include("models/model-exp.jl")
include("models/model-sq.jl")
include("models/model-poly.jl")

Random.seed!(5555)

# Return symbols used as model parameters.
function f_true(x; noise=0.)
    y = exp(x[1]/10) * cos(2*x[1])
    if noise > 0.
        y += rand(Distributions.Normal(0., noise))
    end
    return y
end

# experiment settings
const noise = 0.01
const domain_lb = [0.]
const domain_ub = [20.]

# EXAMPLES - - - - - - - -

function run_boss(model, init_X, init_Y, test_X, test_Y, max_iters; kwargs...)
    # run BOSS
    sample_count = 200
    util_opt_multistart = 100

    f_noisy(x) = f_true(x; noise)
    time = @elapsed X, Y, plots, bsf, errs = boss(f_noisy, init_X, init_Y, model, domain_lb, domain_ub;
        sample_count,
        util_opt_multistart,
        INFO=true,
        max_iters,
        test_X,
        test_Y,
        target_err=nothing,
        kwargs...
    )

    return RunResult(time, X, Y, bsf, errs), plots
end

function example()
    # init data
    init_data_size = 6
    X = hcat(rand(Distributions.Uniform(domain_lb[1], domain_ub[1]), init_data_size))
    Y = [f_true(x; noise) for x in eachrow(X)]

    # test data
    test_data_size = 2000
    test_X = LinRange(domain_lb, domain_ub, test_data_size)
    test_Y = [f_true(x) for x in test_X]

    iters = 2
    model = model_lincos()

    return run_boss(model, X, Y, test_X, test_Y, iters, plot_all_models=true)
end

function compare_models(; save_run_data=false, filename="rundata.jld2")
    # test data
    test_data_size = 2000
    test_X = LinRange(domain_lb, domain_ub, test_data_size)
    test_Y = [f_true(x) for x in test_X]

    # experiment settings
    runs = 10
    iters = 20
    model = model_lincos()

    results = [RunResult[] for _ in 1:3]
    for i in 1:runs
        print("Run $i/$runs ...")

        # init data
        init_data_size = 1
        X = hcat(rand(Distributions.Uniform(domain_lb[1], domain_ub[1]), init_data_size))
        Y = [f_true(x; noise) for x in eachrow(X)]

        param_res, _ = run_boss(model, X, Y, test_X, test_Y, iters; use_model=:param, make_plots=false, info=false)
        semiparam_res, _ = run_boss(model, X, Y, test_X, test_Y, iters, use_model=:semiparam, make_plots=false, info=false)
        nonparam_res, _ = run_boss(model, X, Y, test_X, test_Y, iters; use_model=:nonparam, make_plots=false, info=false)
        push!(results[1], param_res)
        push!(results[2], semiparam_res)
        push!(results[3], nonparam_res)

        print(" done\n")
    end

    labels = ["param", "semiparam", "nonparam"]
    plot_bsf_boxplots(results; labels)

    save_run_data && save_data(results, "example/data/", filename)
    return results
end
