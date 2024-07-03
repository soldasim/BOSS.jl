
"""
Creates a dummy problem for the given combinatorial test.

# Params
- `in::Function`: Maps input variable names to their values.
"""
function create_problem(val)
    problem = construct_problem(val)
    model_fitter = construct_model_fitter(Val(val("ModelFitter")), val)
    acq_maximizer = construct_acq_maximizer(Val(val("AcquisitionMaximizer")), val, problem)
    acquisition = val("Acquisition")
    options = BOSS.BossOptions(; info=true, debug=true)
    term_cond = BOSS.IterLimit(val("iter_max"))

    if ismissing(problem.f)
        return () -> bo!(problem;
            model_fitter,
            acq_maximizer,
            acquisition,
            options,
        )
    else
        return () -> bo!(problem;
            model_fitter,
            acq_maximizer,
            acquisition,
            term_cond,
            options,
        )
    end
end

function construct_problem(val)
    fitness = construct_fitness(Val(val("FITNESS")), val)

    domain = BOSS.Domain(;
        bounds = val("bounds"),
        discrete = val("discrete"),
        cons = val("cons"),
    )
    model = construct_model(Val(val("MODEL")), val)

    data = BOSS.ExperimentDataPrior(val("XY")...)
    
    return BOSS.BossProblem(;
        fitness,
        f = val("f"),
        domain,
        model,
        noise_std_priors = val("noise_std_priors"),
        y_max = val("y_max"),
        data,
    )    
end

function construct_fitness(::Val{:LinFitness}, val)
    return BOSS.LinFitness(val("LinFitness_coefs"))
end
function construct_fitness(::Val{:NonlinFitness}, val)
    return BOSS.NonlinFitness(val("NonlinFitness_fit"))
end

function construct_model(::Val{:Parametric}, val)
    return BOSS.NonlinModel(;
        predict = val("Parametric_predict"),
        param_priors = val("Parametric_theta_priors"),
    )
end
function construct_model(::Val{:Nonparametric}, val)
    return BOSS.Nonparametric(;
        mean = val(String(val("MODEL"))*"_mean"),
        kernel = val("Nonparametric_kernel"),
        amp_priors = val("Nonparametric_amp_priors"),
        length_scale_priors = val("Nonparametric_length_scale_priors"),
    )
end
function construct_model(::Val{:Semiparametric}, val)
    return BOSS.Semiparametric(;
        parametric = construct_model(Val(:Parametric), val),
        nonparametric = construct_model(Val(:Nonparametric), val),
    )
end

function construct_model_fitter(::Val{:Optimization}, val)
    return BOSS.OptimizationMLE(;
        algorithm = NEWUOA(),
        multistart = 2,#200,  # low to improve test runtime
        parallel = val("ModelFitter_parallel"),
        rhoend = 1e-2,
    )
end
function construct_model_fitter(::Val{:Turing}, val)
    # low sample count to improve test runtime
    return BOSS.TuringBI(;
        sampler = BOSS.PG(20),
        warmup = 10,#100,
        samples_in_chain = 2,#10,
        chain_count = 2,#8,
        leap_size = 3,#5,
        parallel = val("ModelFitter_parallel"),
    )
end
function construct_model_fitter(::Val{:Sampling}, val)
    return BOSS.SamplingMLE(;
        samples = 2,#200,  # low to improve test runtime
        parallel = val("ModelFitter_parallel"),
    )
end
function construct_model_fitter(::Val{:Random}, val)
    return BOSS.RandomMLE()
end

function construct_acq_maximizer(::Val{:Optimization}, val, problem)
    return BOSS.OptimizationAM(;
        algorithm = isnothing(val("cons")) ? BOBYQA() : COBYLA(),
        multistart = 2,#20,  # low to improve test runtime
        parallel = val("AcquisitionMaximizer_parallel"),
        rhoend = 1e-2,
    )
end
function construct_acq_maximizer(::Val{:Grid}, val, problem)
    return BOSS.GridAM(;
        problem,
        steps = [1.],#[0.1],  # high to improve test runtime
        parallel = val("AcquisitionMaximizer_parallel"),
    )
end
function construct_acq_maximizer(::Val{:Random}, val, problem)
    return BOSS.RandomAM()
end
