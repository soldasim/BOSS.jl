using Random
using Distributions
using PyCall
using LatinHypercubeSampling

include("../boss.jl")
include("../example/data.jl")

Random.seed!(5555)


# THE OBJECTIVE FUNCTION
@pyinclude("motor/computational_model.py")
function fg_true(x)
    ys, zs = py"calc"(x...)
    return ys, zs
end

fitness(; alpha=1., beta=1.) = LinFitness([1., alpha, beta])

# MODEL
include("motor_model.jl")

# EXPERIMENT SETTINGS
domain_lb() = [20.1, 0.011, 0.2971] #[20., 0.01, 0.297]
domain_ub() = [59.9, 0.029, 0.3999] #[60., 0.03, 0.400]

# RUN BOSS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# main function
function example(max_iters; info=true, kwargs...)
    info && print("generating init data ...\n")
    X, Y, Z = generate_init_data_LHC_(19)
    info && print("generating test data ...\n")
    test_X, test_Y = generate_test_data_(20)

    Z = nothing  # TODO remove
    res = run_boss_(X, Y, Z; max_iters, info, test_X, test_Y, kwargs...)
    return res, (test_X, test_Y)
end

function example_continue(max_iters, res; test_data=(nothing, nothing), info=true, kwargs...)
    res_new = run_boss_(res.X, res.Y, res.Z; max_iters, info, test_X=test_data[1], test_Y=test_data[2], kwargs...)

    return RunResult(
        res.time + res_new.time,
        res_new.X,
        res_new.Y,
        res_new.Z,
        vcat(res.bsf, res_new.bsf[2:end]),
        (isnothing(res.errs) || isnothing(res_new.errs)) ? nothing : vcat(res.errs, res_new.errs[2:end]),
    ), test_data
end

function compare_models(; info=false, save_run_data=false, filename="rundata", make_plots=false)
    # generate data
    test_X, test_Y = generate_test_data_(20)

    # experiment settings
    runs = 10
    max_iters = 200
    init_data_size = 19

    print("Starting $runs runs.\n")
    results = [Vector{RunResult}(undef, runs) for _ in 1:3]
    for i in 1:runs
        print("Thread $(Threads.threadid()):  Run $i in progress ...\n")

        X, Y, Z = generate_init_data_LHC_(init_data_size)
        Z = nothing  # TODO remove

        param_res = run_boss_(X, Y, Z; use_model=:param, max_iters, info, test_X, test_Y, make_plots)
        semiparam_res = run_boss_(X, Y, Z; use_model=:semiparam, max_iters, info, test_X, test_Y, make_plots)
        nonparam_res = run_boss_(X, Y, Z; use_model=:nonparam, max_iters, info, test_X, test_Y, make_plots)

        results[1][i] = param_res
        results[2][i] = semiparam_res
        results[3][i] = nonparam_res

        save_run_data && save_data((param_res, semiparam_res, nonparam_res), "motor/data/", filename*"$i.jld2")
    end

    labels = ["param", "semiparam", "nonparam"]
    plot_bsf_boxplots(results; labels)

    return results
end

# wrapper function
function run_boss_(init_X, init_Y, init_Z; kwargs...)
    util_opt_multistart = 100

    noise_pred = [0.01, 0.01, 0.01]
    constraint_noise = [0.01, 0.01, 0.01]

    fit = fitness(; alpha=(10.)^1, beta=(10.)^5)
    model = motor_model()

    time = @elapsed X, Y, Z, bsf, errs, _ = boss(fg_true, fit, init_X, init_Y, init_Z, model, domain_lb(), domain_ub();
        f_true=x->fg_true(x)[1],
        noise=noise_pred,
        constraint_noise,
        util_opt_multistart,
        target_err=nothing,
        kwargs...
    )

    return RunResult(time, X, Y, Z, bsf, errs)
end

function generate_init_data_random_(size)
    lb, ub = domain_lb(), domain_ub()
    X = hcat(rand(Product(Distributions.Uniform.(lb, ub)), size))'
    return generate_init_data_(X, size)
end

function generate_init_data_LHC_(size)
    lb, ub = domain_lb(), domain_ub()
    X = scaleLHC(randomLHC(size, 3), [(lb[i], ub[i]) for i in 1:3])
    return generate_init_data_(X, size)
end

function generate_init_data_(X, size)
    Y = zeros(size, 3)
    Z = zeros(size, 1)
    for i in 1:size
        y, z = fg_true(X[i,:])
        Y[i,:] .= y
        Z[i,:] .= z
    end
    return X, Y, Z
end

function generate_test_data_(size_per_dim)
    test_X = reduce(hcat, [[x...] for x in collect(Iterators.product([collect(LinRange(domain_lb()[i], domain_ub()[i], size_per_dim)) for i in 1:length(domain_lb())]...))])'
    test_Y = reduce(hcat, [fg_true(x)[1] for x in eachrow(test_X)])'
    return test_X, test_Y
end
