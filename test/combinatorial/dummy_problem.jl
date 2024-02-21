
function create_problem(in)
    problem = construct_problem(in)
    model_fitter = construct_model_fitter(Val(in("ModelFitter")), in)
    acq_maximizer = construct_acq_maximizer(Val(in("AcquisitionMaximizer")), in, problem)
    acquisition = in("Acquisition")
    options = BOSS.BossOptions(; info=false, debug=true)
    term_cond = BOSS.IterLimit(in("iter_max"))

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

function construct_problem(in)
    fitness = construct_fitness(Val(in("FITNESS")), in)

    domain = BOSS.Domain(;
        bounds = in("bounds"),
        discrete = in("discrete"),
        cons = in("cons"),
    )
    model = construct_model(Val(in("MODEL")), in)

    data = BOSS.ExperimentDataPrior(in("XY")...)
    
    return BOSS.OptimizationProblem(;
        fitness,
        f = in("f"),
        domain,
        model,
        noise_var_priors = in("noise_var_priors"),
        y_max = in("y_max"),
        data,
    )    
end

function construct_fitness(::Val{:LinFitness}, in)
    return BOSS.LinFitness(in("LinFitness_coefs"))
end
function construct_fitness(::Val{:NonlinFitness}, in)
    return BOSS.NonlinFitness(in("NonlinFitness_fit"))
end

function construct_model(::Val{:Parametric}, in)
    return BOSS.NonlinModel(;
        predict = in("Parametric_predict"),
        param_priors = in("Parametric_theta_priors"),
    )
end
function construct_model(::Val{:Nonparametric}, in)
    return BOSS.Nonparametric(;
        mean = in(String(in("MODEL"))*"_mean"),
        kernel = in("Nonparametric_kernel"),
        length_scale_priors = in("Nonparametric_length_scale_priors"),
    )
end
function construct_model(::Val{:Semiparametric}, in)
    return BOSS.Semiparametric(;
        parametric = construct_model(Val(:Parametric), in),
        nonparametric = construct_model(Val(:Nonparametric), in),
    )
end

function construct_model_fitter(::Val{:Optimization}, in)
    return BOSS.OptimizationMLE(;
        algorithm = NEWUOA(),
        multistart = 2,#200,  # low to improve test runtime
        parallel = in("ModelFitter_parallel"),
        rhoend = 1e-2,
    )
end
function construct_model_fitter(::Val{:Turing}, in)
    # low sample count to improve test runtime
    return BOSS.TuringBI(;
        sampler = BOSS.PG(20),
        warmup = 10,#100,
        samples_in_chain = 2,#10,
        chain_count = 2,#8,
        leap_size = 3,#5,
        parallel = in("ModelFitter_parallel"),
    )
end
function construct_model_fitter(::Val{:Sampling}, in)
    return BOSS.SamplingMLE(;
        samples = 2,#200,  # low to improve test runtime
        parallel = in("ModelFitter_parallel"),
    )
end
function construct_model_fitter(::Val{:Random}, in)
    return BOSS.RandomMLE()
end

function construct_acq_maximizer(::Val{:Optimization}, in, problem)
    return BOSS.OptimizationAM(;
        algorithm = isnothing(in("cons")) ? BOBYQA() : COBYLA(),
        multistart = 2,#20,  # low to improve test runtime
        parallel = in("AcquisitionMaximizer_parallel"),
        rhoend = 1e-2,
    )
end
function construct_acq_maximizer(::Val{:Grid}, in, problem)
    return BOSS.GridAM(;
        problem,
        steps = [1.],#[0.1],  # high to improve test runtime
        parallel = in("AcquisitionMaximizer_parallel"),
    )
end
function construct_acq_maximizer(::Val{:Random}, in, problem)
    return BOSS.RandomAM()
end
