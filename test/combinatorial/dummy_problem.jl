
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
    options = BossOptions(; info=true, debug=true)
    term_cond = IterLimit(val("iter_max"))

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

    domain = Domain(;
        bounds = val("bounds"),
        discrete = val("discrete"),
        cons = val("cons"),
    )
    model = construct_model(Val(val("MODEL")), val)

    data = ExperimentData(val("XY")...)
    
    return BossProblem(;
        fitness,
        f = val("f"),
        domain,
        model,
        y_max = val("y_max"),
        data,
    )    
end

function construct_fitness(::Val{:LinFitness}, val)
    return LinFitness(val("LinFitness_coefs"))
end
function construct_fitness(::Val{:NonlinFitness}, val)
    return NonlinFitness(val("NonlinFitness_fit"))
end

function construct_model(::Val{:Parametric}, val; no_noise=false)
    return NonlinearModel(;
        predict = val("Parametric_predict"),
        theta_priors = val("Parametric_theta_priors"),
        noise_std_priors = no_noise ? nothing : val("noise_std_priors"),
    )
end
function construct_model(::Val{:Nonparametric}, val)
    return Nonparametric(;
        mean = val(String(val("MODEL"))*"_mean"),
        kernel = val("Nonparametric_kernel"),
        amplitude_priors = val("Nonparametric_amplitude_priors"),
        lengthscale_priors = val("Nonparametric_lengthscale_priors"),
        noise_std_priors = val("noise_std_priors"),
    )
end
function construct_model(::Val{:Semiparametric}, val)
    return Semiparametric(;
        parametric = construct_model(Val(:Parametric), val; no_noise=true),
        nonparametric = construct_model(Val(:Nonparametric), val),
    )
end

function construct_model_fitter(::Val{:Sampling}, val)
    return SamplingMAP(;
        samples = 10,#200,  # low to improve test runtime
        parallel = val("ModelFitter_parallel"),
    )
end
function construct_model_fitter(::Val{:Optimization}, val)
    return OptimizationMAP(;
        algorithm = NEWUOA(),
        multistart = 2,#200,  # low to improve test runtime
        parallel = val("ModelFitter_parallel"),
        rhoend = 1e-2,
    )
end
# function construct_model_fitter(::Val{:SampleOpt}, val)
#     return SampleOptMAP(;
#         samples = 10,#200,  # low to improve test runtime
#         algorithm = NEWUOA(),
#         multistart = 2,#200,  # low to improve test runtime
#         parallel = val("ModelFitter_parallel"),
#         rhoend = 1e-2,
#     )
# end
function construct_model_fitter(::Val{:Turing}, val)
    # low sample count to improve test runtime
    return TuringBI(;
        sampler = PG(20),
        warmup = 10,#100,
        samples_in_chain = 2,#10,
        chain_count = 2,#8,
        leap_size = 3,#5,
        parallel = val("ModelFitter_parallel"),
    )
end
function construct_model_fitter(::Val{:Random}, val)
    return RandomFitter()
end

function construct_acq_maximizer(::Val{:Sampling}, val, problem)
    return SamplingAM(;
        x_prior = product_distribution(Uniform.(val("bounds")...)),
        samples = 10,#200,  # low to improve test runtime
        parallel = val("AcquisitionMaximizer_parallel"),
    )
end
function construct_acq_maximizer(::Val{:Optimization}, val, problem)
    return OptimizationAM(;
        algorithm = isnothing(val("cons")) ? BOBYQA() : COBYLA(),
        multistart = 2,#20,  # low to improve test runtime
        parallel = val("AcquisitionMaximizer_parallel"),
        rhoend = 1e-2,
    )
end
# function construct_acq_maximizer(::Val{:SampleOpt}, val, problem)
#     return SampleOptAM(;
#         x_prior = product_distribution(Uniform.(val("bounds")...)),
#         samples = 10,#200,  # low to improve test runtime
#         algorithm = isnothing(val("cons")) ? BOBYQA() : COBYLA(),
#         multistart = 2,#20,  # low to improve test runtime
#         parallel = val("AcquisitionMaximizer_parallel"),
#         rhoend = 1e-2,
#     )
# end
function construct_acq_maximizer(::Val{:Grid}, val, problem)
    return GridAM(;
        problem,
        steps = [1.],#[0.1],  # high to improve test runtime
        parallel = val("AcquisitionMaximizer_parallel"),
    )
end
function construct_acq_maximizer(::Val{:Random}, val, problem)
    return RandomAM()
end
