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

# The objective function.
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

function run_boss(model, init_X, init_Y, max_iters, make_plots=true; kwargs...)
    sample_count = 200
    util_opt_multistart = 100

    f_noisy(x) = f_true(x; noise)
    time = @elapsed X, Y, plots, bsf, errs = boss(f_noisy, init_X, init_Y, model, domain_lb, domain_ub;
        sample_count,
        util_opt_multistart,
        INFO=true,
        make_plots,
        max_iters,
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

    test_X, test_Y = generate_test_data()

    iters = 2
    model = model_lincos()

    return run_boss(model, X, Y, iters, make_plots=true, plot_all_models=true; test_X, test_Y)
end

function compare_models(; save_run_data=false, filename="rundata.jld2", make_plots=true)
    # init data
    init_data_size = 2
    X = hcat(rand(Distributions.Uniform(domain_lb[1], domain_ub[1]), init_data_size))
    Y = [f_true(x; noise) for x in eachrow(X)]

    test_X, test_Y = generate_test_data()

    # experiment settings
    runs = 16
    iters = 20
    model = model_lincos()

    results = [Vector{RunResult}(undef, runs) for _ in 1:3]
    for i in 1:runs # TODO Threads.@threads
        # print("Run $i/$runs ... (thread $(Threads.threadid()))\n")

        param_res, _ = run_boss(model, X, Y, iters; test_X, test_Y, use_model=:param, make_plots, info=false)
        semiparam_res, _ = run_boss(model, X, Y, iters; test_X, test_Y, use_model=:semiparam, make_plots, info=false)
        nonparam_res, _ = run_boss(model, X, Y, iters; test_X, test_Y, use_model=:nonparam, make_plots, info=false)

        results[1][i] = param_res
        results[2][i] = semiparam_res
        results[3][i] = nonparam_res
    end

    labels = ["param", "semiparam", "nonparam"]
    plot_bsf_boxplots(results; labels)

    save_run_data && save_data(results, "example/data/", filename)
    return results
end

function generate_test_data()
    test_data_size = 2000
    test_X = reduce(hcat, (LinRange(domain_lb, domain_ub, test_data_size)))'
    test_Y = [f_true(x) for x in eachrow(test_X)]
    return test_X, test_Y
end
