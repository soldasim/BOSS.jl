using Random
using Distributions
using PyCall
using LatinHypercubeSampling
using KernelFunctions

include("../src/boss.jl")
include("../example/data.jl")

Random.seed!(5555)


# THE OBJECTIVE FUNCTION
@pyinclude("motor/computational_model.py")
function obj_func(x)
    ys = py"calc"(x...)
    return ys
end

fitness(; alpha=1., beta=1.) = Boss.LinFitness([1., alpha, beta])

# MODEL
include("motor_model.jl")

# DOMAIN CONSTRAINTS
# domain_bounds() = [20., 0.01, 0.297], [60., 0.03, 0.400]
domain_bounds() = [20.1, 0.011, 0.2971], [59.9, 0.029, 0.3999]

function domain_constraints()
    # bounds
    lb, ub = domain_bounds()

    # constants
    duct_gap = 0.005
    D1 = 0.297
    D1_gap = 0.002
    D2 = 0.4
    D2_gap = 0.003

    # constraints
    lc, uc = [0.], [Inf]

    function constraints(x)
        nk, dk, Ds = x
        
        const_gap = nk * (dk + duct_gap) / pi
        const_D1 = Ds - dk/2
        const_D2 = Ds + dk/2
        
        const_1 = const_gap - Ds
        const_2 = (D1 + 2 * D1_gap) - const_D1
        const_3 = const_D2 - (D2 - 2 * D2_gap)

        return [const_1, const_2, const_3]
    end

    lc = [-Inf, -Inf, -Inf]
    uc = [0., 0., 0.]

    function cons_c!(c, x)
        c_ = constraints(x)
        for i in eachindex(c_)
            c[i] = c_[i]
        end
    end

    return TwiceDifferentiableConstraints(cons_c!, lb, ub, lc, uc, :forward)
end

# RUN BOSS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# main function
function example(max_iters; info=true, kwargs...)
    info && print("generating init data ...\n")
    X, Y = generate_init_data_LHC_(19)
    info && print("generating test data ...\n")
    # test_X, test_Y = generate_test_data_(20)

    res = run_boss_(X, Y; max_iters, info, kwargs...) # test_X, test_Y
    return res #, (test_X, test_Y)
end

function example_continue(max_iters, res; test_data=(nothing, nothing), info=true, kwargs...)
    res_new = run_boss_(res.X, res.Y; max_iters, info, test_X=test_data[1], test_Y=test_data[2], kwargs...)

    return RunResult(
        res.time + res_new.time,
        res_new.X,
        res_new.Y,
        res_new.Z,
        vcat(res.bsf, res_new.bsf[2:end]),
        (isnothing(res.errs) || isnothing(res_new.errs)) ? nothing : vcat(res.errs, res_new.errs[2:end]),
    ), test_data
end

function compare_models(; info=false, save_run_data=false, filename="rundata", make_plots=false, kwargs...)
    # generate data
    # test_X, test_Y = generate_test_data_(20)

    # experiment settings
    runs = 10
    max_iters = 200
    init_data_size = 19

    print("Starting $runs runs.\n")
    results = [Vector{RunResult}(undef, runs) for _ in 1:3]
    for i in 1:runs
        print("Thread $(Threads.threadid()):  Run $i in progress ...\n")

        X, Y = generate_init_data_LHC_(init_data_size)

        param_res = run_boss_(X, Y; use_model=:param, max_iters, info, make_plots, kwargs...)
        semiparam_res = run_boss_(X, Y; use_model=:semiparam, max_iters, info, make_plots, kwargs...)
        nonparam_res = run_boss_(X, Y; use_model=:nonparam, max_iters, info, make_plots, kwargs...)

        results[1][i] = param_res
        results[2][i] = semiparam_res
        results[3][i] = nonparam_res

        if save_run_data
            try
                save_data((param_res, semiparam_res, nonparam_res), "motor/data/", filename*"$i.jld2")
            catch e
                showerror(stdout, e)
            end
        end
    end

    try
        labels = ["param", "semiparam", "nonparam"]
        plot_bsf_boxplots(results; labels)
    catch e
        showerror(stdout, e)
    end

    return results
end

# wrapper function
function run_boss_(init_X, init_Y; kwargs...)
    mc_settings = Boss.MCSettings(50, 10, 12, 3)
    vi_samples = 200
    param_fit_alg = :VI
    param_opt_multistart = 24
    acq_opt_multistart = 24

    noise_priors = [LogNormal(-2.3, 1.) for _ in 1:3]

    fit = fitness(; alpha=(10.)^1, beta=(10.)^5)
    model = motor_model()
    domain = domain_constraints()
    kernel = Boss.DiscreteKernel(Matern52Kernel(), [true, false, false])

    time = @elapsed X, Y, bsf, errs, _ = Boss.boss(obj_func, fit, init_X, init_Y, model, domain;
        noise_priors,
        mc_settings,
        acq_opt_multistart,
        param_opt_multistart,
        target_err=nothing,
        param_fit_alg,
        vi_samples,
        kernel,
        kwargs...
    )

    return RunResult(time, X, Y, nothing, bsf, errs)
end

function generate_init_data_random_(size)
    lb, ub = domain_lb(), domain_ub()
    X = hcat(rand(Product(Distributions.Uniform.(lb, ub)), size))'
    return generate_init_data_(X, size)
end

function generate_init_data_LHC_(size)
    lb, ub = domain_bounds()
    X = scaleLHC(randomLHC(size, 3), [(lb[i], ub[i]) for i in 1:3])
    return generate_init_data_(X, size)
end

function generate_init_data_(X, size)
    Y = zeros(size, 3)
    for i in 1:size
        y = obj_func(X[i,:])
        Y[i,:] .= y
    end
    return X, Y
end

function generate_test_data_(size_per_dim)
    test_X = reduce(hcat, [[x...] for x in collect(Iterators.product([collect(LinRange(domain_lb()[i], domain_ub()[i], size_per_dim)) for i in 1:length(domain_lb())]...))])'
    test_Y = reduce(hcat, [obj_func(x)[1] for x in eachrow(test_X)])'
    return test_X, test_Y
end
