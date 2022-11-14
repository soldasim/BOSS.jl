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
function obj_func(x; noise=0.)
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
    g1 = 5. - x[1] + rand(distr)
    return [g1]
end

noise_real() = 0.1
noise_prior() = LogNormal(-2.3, 0.5)

# EXAMPLES - - - - - - - -

function example(max_iters; init_data_size=2, info=true, make_plots=true, plot_all_models=true, kwargs...)
    # Random.seed!(5555)
    
    X, Y = generate_init_data_(init_data_size; noise=noise_real(), feasibility=true)
    # X, Y = generate_init_data_(init_data_size; noise=noise_real(), feasibility=false)

    # test_X, test_Y = generate_test_data_(2000)
    test_X, test_Y = nothing, nothing

    lincos = model_lincos()
    # model = lincos
    model = Boss.NonlinModel(
        (x,ps) -> [lincos.predict(x,ps)..., 0.],
        lincos.param_priors,
        lincos.param_count,
    )

    return run_boss_(model, X, Y; max_iters, info, make_plots, plot_all_models, test_X, test_Y, kwargs...)
end

function compare_models(; save_run_data=false, filename="rundata.jld2", make_plots=false, kwargs...)
    test_X, test_Y = generate_test_data_(2000)

    # experiment settings
    runs = 10
    max_iters = 20
    
    lincos = model_lincos()
    # model = lincos
    model = Boss.NonlinModel(
        (x,ps) -> [lincos.predict(x,ps)..., 0.],
        lincos.param_priors,
        lincos.param_count,
    )

    results = [Vector{RunResult}(undef, runs) for _ in 1:3]
    for i in 1:runs
        print("\n ### RUN $i/$runs ### \n")
        info = true
        
        X, Y = generate_init_data_(2; noise=noise_real(), feasibility=true)

        param_res, _ = run_boss_(model, X, Y; max_iters, test_X, test_Y, use_model=:param, make_plots, info, kwargs...)
        semiparam_res, _ = run_boss_(model, X, Y; max_iters, test_X, test_Y, use_model=:semiparam, make_plots, info, kwargs...)
        nonparam_res, _ = run_boss_(model, X, Y; max_iters, test_X, test_Y, use_model=:nonparam, make_plots, info, kwargs...)

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

function run_boss_(model, init_X, init_Y; kwargs...)
    mc_settings = Boss.MCSettings(50, 6, 8, 3)
    acq_opt_multistart = 12
    param_opt_multistart = 20

    f_true(x) = vcat(obj_func(x), feasibility_constraints(x))

    # f(x; noise=0.) = f_true(x; noise)
    f(x; noise=0.) = vcat(obj_func(x; noise), feasibility_constraints(x; noise))
    f_noisy(x) = f(x; noise=noise_real())

    # constraints = nothing
    constraints = [Inf, 0.]
    
    # domain = ([0.], [20.])
    domain = domain_constraints()
    # domain = evolutionary_constraints()
    
    # fitness = Boss.LinFitness([1.])
    fitness = Boss.LinFitness([1., 0.])

    param_fit_alg = :BI #:MLE
    acq_opt_alg = :Optim #:CMAES
    kernel = Matern52Kernel()
    # kernel = Boss.DiscreteKernel(Matern52Kernel())

    # noise_priors = [noise_prior()]
    # gp_params_priors = [Product([Gamma(2., 1.)])]
    noise_priors = [noise_prior() for _ in 1:2]
    gp_params_priors = [Product([Gamma(2., 1.)]) for _ in 1:2]

    time = @elapsed X, Y, bsf, parameters, errs, plots = Boss.boss(f_noisy, fitness, init_X, init_Y, model, domain;
        f_true,
        constraints,
        noise_priors,
        mc_settings,
        acq_opt_multistart,
        param_opt_multistart,
        target_err=nothing,
        gp_params_priors,
        param_fit_alg,
        acq_opt_alg,
        kernel,
        parallel=false,
        kwargs...
    )

    return RunResult(time, X, Y, constraints, bsf, parameters, errs), plots
end

function generate_init_data_(size; noise=0., feasibility)
    X = rand(Product(Distributions.Uniform.(domain_bounds()...)), size)
    Y = reduce(hcat, obj_func.(eachcol(X); noise))
    if feasibility
        Z = reduce(hcat, feasibility_constraints.(eachcol(X); noise))
        Y = vcat(Y, Z)
    end
    return X, Y
end

function generate_test_data_(size)
    test_X = reduce(hcat, (LinRange(domain_bounds()..., size)))
    test_Y = reduce(hcat, [obj_func(x) for x in eachrow(test_X)])
    return test_X, test_Y
end
