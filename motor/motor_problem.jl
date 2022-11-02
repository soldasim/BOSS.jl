using Random
using Distributions
using PyCall
using LatinHypercubeSampling
using KernelFunctions
using Optim
using Evolutionary

include("../src/boss.jl")
include("../example/data.jl")


## OPT VERSION

# THE OBJECTIVE FUNCTION
@pyinclude("motor/computational_model.py")
function obj_func(x)
    ys = py"calc"(x...)
    return ys
end

fitness(; alpha=1., beta=1.) = Boss.LinFitness([1., alpha, beta])

# MODEL
include("motor_model.jl")


# ### COEFF VERSION
# # For some reason crashes with parallelization.
# # Add `parallel=false` kwarg to boss.

# # THE OBJECTIVE FUNCTION
# @pyinclude("motor/main_coeff.py")
# function obj_func(x, coef=0.8)
#     ys = py"calc"(x..., coef)
#     return ys
# end

# fitness(; alpha=1., beta=1.) = Boss.LinFitness([1., alpha, beta])

# # MODEL
# motor_model() = Boss.NonlinModel(
#     (x, params) -> obj_func(x, params[1]),
#     [Distributions.Uniform(0., 1.)],
#     1,
# )


# DOMAIN CONSTRAINTS
# domain_bounds() = [20., 0.01, 0.297], [60., 0.03, 0.400]
domain_bounds() = [20.1, 0.011, 0.2971], [59.9, 0.029, 0.3999]

# rewritten according to the motor source code ('computational_model.py')
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

    # return MixedTypePenaltyConstraints(
    #     WorstFitnessConstraints(lb, ub, lc, uc, constraints),
    #     [Int, Float64, Float64],
    # )

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
function example(max_iters; info=true, init_data_size=19, kwargs...)
    Random.seed!(5555)

    info && print("generating init data ...\n")
    X, Y = generate_init_data_LHC_(init_data_size)
    # info && print("generating test data ...\n")
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

function compare_models(; save_run_data=false, file="motor/data/rundata", make_plots=false, make_boxplot=false, kwargs...)
    # generate data
    # test_X, test_Y = generate_test_data_(20)

    # experiment settings
    runs = 10
    max_iters = 100
    init_data_size = 19

    print("Starting $runs runs.\n")
    results = [Vector{RunResult}(undef, runs) for _ in 1:3]
    for i in 1:runs
        print("\n ### RUN $i/$runs ### \n")
        info = true

        X, Y = generate_init_data_LHC_(init_data_size)

        param_res = run_boss_(X, Y; use_model=:param, max_iters, info, make_plots, kwargs...)
        if save_run_data
            try
                save_data(param_res, "./", file*"_param_$i.jld2")
            catch e
                showerror(stdout, e)
            end
        end

        semiparam_res = run_boss_(X, Y; use_model=:semiparam, max_iters, info, make_plots, kwargs...)
        if save_run_data
            try
                save_data(param_res, "./", file*"_semiparam_$i.jld2")
            catch e
                showerror(stdout, e)
            end
        end

        nonparam_res = run_boss_(X, Y; use_model=:nonparam, max_iters, info, make_plots, kwargs...)
        if save_run_data
            try
                save_data(param_res, "./", file*"_nonparam_$i.jld2")
            catch e
                showerror(stdout, e)
            end
        end

        results[1][i] = param_res
        results[2][i] = semiparam_res
        results[3][i] = nonparam_res

        # if save_run_data
        #     try
        #         save_data((param_res, semiparam_res, nonparam_res), "./", file*"_$i.jld2")
        #     catch e
        #         showerror(stdout, e)
        #     end
        # end
    end

    if make_boxplot
        try
            labels = ["param", "semiparam", "nonparam"]
            plot_bsf_boxplots(results; labels)
        catch e
            showerror(stdout, e)
        end
    end

    return results
end

# wrapper function
function run_boss_(init_X, init_Y; kwargs...)
    mc_settings = Boss.MCSettings(50, 8, 6, 3)
    param_fit_alg = :BI
    acq_opt_multistart = 12

    noise_priors = [LogNormal(-2.3, 1.) for _ in 1:3]

    fit = fitness(; alpha=(10.)^1, beta=(10.)^5)
    model = motor_model()
    domain = domain_constraints()
    kernel = Boss.DiscreteKernel(Matern52Kernel(), [true, false, false])
    
    acq_opt_alg = :Optim
    optim_options = Optim.Options(; x_abstol=1e-2, iterations=800)
    cmaes_options = Evolutionary.Options(; abstol=1e-0, iterations=800)

    time = @elapsed X, Y, bsf, errs, _ = Boss.boss(obj_func, fit, init_X, init_Y, model, domain;
        noise_priors,
        mc_settings,
        acq_opt_multistart,
        target_err=nothing,
        param_fit_alg,
        kernel,
        acq_opt_alg,
        optim_options,
        cmaes_options,
        # parallel=false,
        kwargs...
    )

    return RunResult(time, X, Y, nothing, bsf, errs)
end

function generate_init_data_random_(size)
    lb, ub = domain_lb(), domain_ub()
    X = hcat(rand(Product(Distributions.Uniform.(lb, ub)), size))
    return generate_init_data_(X, size)
end

function generate_init_data_LHC_(size)
    lb, ub = domain_bounds()
    X = scaleLHC(randomLHC(size, 3), [(lb[i], ub[i]) for i in 1:3])'
    return generate_init_data_(X, size)
end

function generate_init_data_(X, size)
    Y = zeros(3, size)
    for i in 1:size
        y = obj_func(X[:,i])
        Y[:,i] .= y
    end
    return X, Y
end

function generate_test_data_(size_per_dim)
    test_X = reduce(hcat, [[x...] for x in collect(Iterators.product([collect(LinRange(domain_lb()[i], domain_ub()[i], size_per_dim)) for i in 1:length(domain_lb())]...))])
    test_Y = reduce(hcat, [obj_func(x)[1] for x in eachrow(test_X)])
    return test_X, test_Y
end
