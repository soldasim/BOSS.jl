using Random
using Distributions
using Optim
using AbstractGPs

# TODO example env ?
# TODO: using .Boss

include("../src/boss.jl")
include("data.jl")
include("../src/plotting.jl")

# Load model
include("models/model-expcos.jl")
include("models/model-lincos.jl")
include("models/model-poly.jl")

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

function evolutionary_constraints()
    # bounds
    lb, ub = domain_bounds()

    # constraints
    lc = [0.]
    uc = [Inf]
    c(x) = [-x[1] + 15.]

    return Boss.Evolutionary.WorstFitnessConstraints(lb, ub, lc, uc, c)
end

# Feasibility constraints on output. (x > 5.)
function feasibility_constraints(x; noise=0.)
    distr = Distributions.Normal(0., noise)
    g1 = x[1] - 5 + rand(distr)
    return [g1]
end

noise_real() = 0.1
noise_prior() = LogNormal(-2.3, 0.5)

# EXAMPLES - - - - - - - -

function example(max_iters; init_data_size=2, info=true, make_plots=true, plot_all_models=true, kwargs...)
    # Random.seed!(5555)
    
    X, Y, Z = generate_init_data_(init_data_size; noise=noise_real(), feasibility=true)
    # X, Y, Z = generate_init_data_(init_data_size; noise=noise_real(), feasibility=false)..., nothing
    # X = [5.;8.;;]; Y = reduce(hcat, [f_true(x; noise=noise_real()) for x in eachrow(X)])'; Z = nothing;

    # test_X, test_Y = generate_test_data_(2000)
    test_X, test_Y = nothing, nothing

    model = model_lincos()
    return run_boss_(model, X, Y, Z; max_iters, info, make_plots, plot_all_models, test_X, test_Y, kwargs...)
end

function compare_models(; save_run_data=false, filename="rundata.jld2", make_plots=false, kwargs...)
    test_X, test_Y = generate_test_data_(2000)

    # experiment settings
    runs = 10
    max_iters = 30
    model = model_lincos()

    results = [Vector{RunResult}(undef, runs) for _ in 1:3]
    for i in 1:runs
        print("\n ### RUN $i/$runs ### \n")
        info = true
        
        X, Y, Z = generate_init_data_(2; noise=noise_real(), feasibility=true)

        param_res, _ = run_boss_(model, X, Y, Z; max_iters, test_X, test_Y, use_model=:param, make_plots, info, kwargs...)
        semiparam_res, _ = run_boss_(model, X, Y, Z; max_iters, test_X, test_Y, use_model=:semiparam, make_plots, info, kwargs...)
        nonparam_res, _ = run_boss_(model, X, Y, Z; max_iters, test_X, test_Y, use_model=:nonparam, make_plots, info, kwargs...)

        results[1][i] = param_res
        results[2][i] = semiparam_res
        results[3][i] = nonparam_res
    end

    try
        plot_bsf_boxplots(results)
    catch e
        showerror(stdout, e)
    end

    if save_run_data
        try
            save_data(results, "example/data/", filename)
        catch e
            showerror(stdout, e)
        end
    end

    return results
end

function run_boss_(model, init_X, init_Y, init_Z; kwargs...)
    mc_settings = Boss.MCSettings(50, 6, 8, 3)
    acq_opt_multistart = 12
    param_opt_multistart = 20

    # fg(x; noise=0.) = f_true(x; noise), feasibility_constraints(x; noise)
    fg(x; noise=0.) = f_true(x; noise), Float64[]
    fg_noisy(x) = fg(x; noise=noise_real())
    
    # domain = ([0.], [20.])
    # domain = domain_constraints()
    domain = evolutionary_constraints()
    
    fitness = Boss.LinFitness([1.])
    # fitness = Boss.NonlinFitness(y -> y[1])
    
    param_fit_alg = :BI
    feasibility_param_fit_alg = :MLE
    acq_opt_alg = :CMAES
    kernel = Matern52Kernel()
    # kernel = Boss.DiscreteKernel(Matern52Kernel())

    noise_priors = [noise_prior()]
    feasibility_noise_priors = [noise_prior()]
    gp_params_priors = [Product([Gamma(2., 1.)])]
    feasibility_gp_params_priors = [Product([Gamma(2., 1.)])]

    time = @elapsed X, Y, Z, bsf, errs, plots = Boss.boss(fg_noisy, fitness, init_X, init_Y, nothing, model, domain;
        f_true,
        noise_priors,
        feasibility_noise_priors,
        mc_settings,
        acq_opt_multistart,
        param_opt_multistart,
        target_err=nothing,
        gp_params_priors,
        feasibility_gp_params_priors,
        param_fit_alg,
        feasibility_param_fit_alg,
        acq_opt_alg,
        kernel,
        kwargs...
    )

    return RunResult(time, X, Y, Z, bsf, errs), plots
end

function generate_init_data_(size; noise=0., feasibility)
    X = rand(Product(Distributions.Uniform.(domain_bounds()...)), size)
    Y = reduce(hcat, f_true.(eachcol(X); noise))
    if feasibility
        Z = reduce(hcat, [feasibility_constraints(x; noise) for x in eachrow(X)])
        return X, Y, Z
    else
        return X, Y
    end
end

function generate_test_data_(size)
    test_X = reduce(hcat, (LinRange(domain_bounds()..., size)))
    test_Y = reduce(hcat, [f_true(x) for x in eachrow(test_X)])
    return test_X, test_Y
end
