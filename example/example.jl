using Random
using Distributions

include("../src/boss.jl")
include("data.jl")

# Load model
include("models/model-expcos.jl")
include("models/model-lincos.jl")
include("models/model-poly.jl")

Random.seed!(5555)

# The objective function.
function f_true(x; noise=0.)
    y = exp(x[1]/10) * cos(2*x[1]) #+ x[2]
    y += rand(Distributions.Normal(0., noise))
    return [y]
end

# The constraints. ( g1(x), g2(x), ... > 0 )
function constraints(x; noise=0.)
    distr = Distributions.Normal(0., noise)
    g1 = x[1] - 5 + rand(distr)
    g2 = - x[1] + 15 + rand(distr)
    return [g1, g2]
end

# experiment settings
noise_real() = 0.1
noise_pred() = 0.1
domain_lb() = [0.] #[0.,0.]
domain_ub() = [20.] #[20.,20.]

# EXAMPLES - - - - - - - -

function example(max_iters; kwargs...)
    # generate data
    X, Y, Z = generate_init_data_(6; noise=noise_real(), constrained=true)
    test_X, test_Y = generate_test_data_(2000)

    # model
    model = model_lincos()

    return run_boss_(model, X, Y, Z; max_iters, info=true, make_plots=true, plot_all_models=true, test_X, test_Y, kwargs...)
end

function compare_models(; save_run_data=false, filename="rundata.jld2", make_plots=false, kwargs...)
    test_X, test_Y = generate_test_data_(2000)

    # experiment settings
    runs = 16
    max_iters = 20
    model = model_lincos()

    print("Starting $runs runs.\n")
    results = [Vector{RunResult}(undef, runs) for _ in 1:3]
    for i in 1:runs  # for parallel computation use macro 'Threads.@threads'
        print("Thread $(Threads.threadid()):  Run $i in progress ...\n")
        info = true
        
        X, Y, Z = generate_init_data_(2; noise=noise_real(), constrained=true)

        param_res, _ = run_boss_(model, X, Y, Z; max_iters, test_X, test_Y, use_model=:param, make_plots, info, kwargs...)
        semiparam_res, _ = run_boss_(model, X, Y, Z; max_iters, test_X, test_Y, use_model=:semiparam, make_plots, info, kwargs...)
        nonparam_res, _ = run_boss_(model, X, Y, Z; max_iters, test_X, test_Y, use_model=:nonparam, make_plots, info, kwargs...)

        results[1][i] = param_res
        results[2][i] = semiparam_res
        results[3][i] = nonparam_res
    end

    labels = ["param", "semiparam", "nonparam"]
    plot_bsf_boxplots(results; labels)

    save_run_data && save_data(results, "example/data/", filename)
    return results
end

function run_boss_(model, init_X, init_Y, init_Z; kwargs...)
    sample_count = 2000
    util_opt_multistart = 100

    fg(x; noise=0.) = f_true(x; noise), constraints(x; noise)
    fg_noisy(x) = fg(x; noise=noise_real())
    fitness = Boss.LinFitness([1.])

    constraint_noise = [noise_pred() for _ in 1:2]

    time = @elapsed X, Y, Z, bsf, errs, plots = Boss.boss(fg_noisy, fitness, init_X, init_Y, init_Z, model, domain_lb(), domain_ub();
        f_true,
        noise=noise_pred(),
        constraint_noise,
        sample_count,
        util_opt_multistart,
        target_err=nothing,
        kwargs...
    )

    return RunResult(time, X, Y, Z, bsf, errs), plots
end

function generate_init_data_(size; noise=0., constrained)
    X = hcat(rand(Product(Distributions.Uniform.(domain_lb(), domain_ub())), size))'
    Y = reduce(hcat, [f_true(x; noise) for x in eachrow(X)])'
    if constrained
        Z = reduce(hcat, [constraints(x; noise) for x in eachrow(X)])'
        return X, Y, Z
    else
        return X, Y
    end
end

function generate_test_data_(size)
    test_X = reduce(hcat, (LinRange(domain_lb(), domain_ub(), size)))'
    test_Y = reduce(hcat, [f_true(x) for x in eachrow(test_X)])'
    return test_X, test_Y
end
