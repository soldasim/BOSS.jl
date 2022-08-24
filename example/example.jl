using Random
using Distributions
using Optim

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

# Domain constraints on input. (x < 15.)
domain_bounds() = [0.], [20.]

function domain_constraints()
    # bounds
    lb, ub = domain_bounds()

    # constraints
    lc = [0.]
    uc = [Inf]
    function cons_c!(c, x)
        c[1] = -x[1] + 15.
        return c
    end
    function cons_jac!(J, x)
        J[1,1] = -1. 
        return J
    end
    function cons_hes!(h, x, λ)
        h[1,1] += λ[1] * 0.
        return h
    end

    return TwiceDifferentiableConstraints(cons_c!, cons_jac!, cons_hes!, lb, ub, lc, uc)
end

# Feasibility constraints on output. (x > 5.)
function feasibility_constraints(x; noise=0.)
    distr = Distributions.Normal(0., noise)
    g1 = x[1] - 5 + rand(distr)
    return [g1]
end

noise_real() = 0.1
noise_prior() = LogNormal(-2.3, 1.)

# EXAMPLES - - - - - - - -

function example(max_iters; kwargs...)
    X, Y, Z = generate_init_data_(2; noise=noise_real(), feasibility=true)
    # test_X, test_Y = generate_test_data_(2000)
    test_X, test_Y = nothing, nothing

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
        
        X, Y, Z = generate_init_data_(2; noise=noise_real(), feasibility=true)

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
    mc_sample_count = 20 #2000
    acq_opt_multistart = 16 #100

    fg(x; noise=0.) = f_true(x; noise), feasibility_constraints(x; noise)
    fg_noisy(x) = fg(x; noise=noise_real())
    
    fitness = Boss.LinFitness([1.])
    # fitness = Boss.NonlinFitness(y -> y[1])
    
    gp_hyperparam_alg = :NUTS
    # gp_hyperparam_alg = :LBFGS

    noise_priors = [noise_prior()]
    feasibility_noise_priors = [noise_prior()]
    gp_params_priors = [MvLogNormal(ones(1), ones(1))]
    feasibility_gp_params_priors = [MvLogNormal(ones(1), ones(1)) for _ in 1:2]

    time = @elapsed X, Y, Z, bsf, errs, plots = Boss.boss(fg_noisy, fitness, init_X, init_Y, init_Z, model, domain_constraints();
        f_true,
        noise_priors,
        feasibility_noise_priors,
        mc_sample_count,
        acq_opt_multistart,
        target_err=nothing,
        gp_params_priors,
        feasibility_gp_params_priors,
        gp_hyperparam_alg,
        kwargs...
    )

    return RunResult(time, X, Y, Z, bsf, errs), plots
end

function generate_init_data_(size; noise=0., feasibility)
    X = hcat(rand(Product(Distributions.Uniform.(domain_bounds()...)), size))'
    Y = reduce(hcat, [f_true(x; noise) for x in eachrow(X)])'
    if feasibility
        Z = reduce(hcat, [feasibility_constraints(x; noise) for x in eachrow(X)])'
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
