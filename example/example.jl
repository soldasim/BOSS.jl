using BOSS
using Distributions
using Optim
using Turing

# The unknown blackbox function.
function blackbox(x)
    y = exp(x[1]/10) * cos(2*x[1])
    z = x[1]^2 - sqrt(15.)
    
    y += rand(Normal(0., 0.5))
    z += rand(Normal(0., 0.5))

    return [y,z]
end

# Our parametric model represents our prediction about the blackbox function.
# 
# Let's assume we know that x->y is a periodic function, so we use `cos` there.
# We don't know anything about x->z, so we put a constant 0 there.
# (That is equivalent to using a simple GP with zero-mean to model x->z.)
function parametric_model()
    y(x, θ) = θ[1] * x[1] * cos(θ[2] * x[1]) + θ[3]
    z(x, θ) = 0.
    predict(x, θ) = [y(x, θ), z(x, θ)]

    θ_priors = fill(Normal(0., 10.), 3)

    BOSS.NonlinModel(predict, θ_priors)
end

# Our prediction about the noise and GP length scales.
noise_var_priors() = fill(Truncated(Normal(0.5, 1.), 0., Inf), 2)
length_scale_priors() = fill(Product([Gamma(2., 1.)]), 2)

# Initial data.
function init_data()
    X = hcat([10.])
    Y = reduce(hcat, blackbox.(eachcol(X)))
    return X, Y
end

"""
We have an unknown noisy function 'blackbox(x)=y,z' and we want to maximize y s.t. z < 0 on domain x ∈ [0,20].
"""
function example()
    problem = BOSS.OptimizationProblem(;
        fitness = BOSS.LinFitness([1, 0]),
        f = blackbox,
        cons = [Inf, 0.],
        domain = ([0.], [20.]),
        discrete = [false],
        model = BOSS.Semiparametric(parametric_model(), BOSS.Nonparametric(; length_scale_priors=length_scale_priors())),
        noise_var_priors = noise_var_priors(),
        data = BOSS.ExperimentDataPrior(init_data()...),
    )

    model_fitter = BOSS.TuringBI(;
        sampler=Turing.PG(20),
        warmup=50,
        samples_in_chain=10,
        chain_count=1,
        leap_size=3,
        parallel=false,
    )
    acq_maximizer = BOSS.OptimMaximizer(;
        algorithm=Fminbox(LBFGS()),
        options=Optim.Options(; x_abstol=0.01),
        multistart=16,
        parallel=true,
    )
    term_cond = BOSS.IterLimit(1)
    options = BOSS.BossOptions(; info=true)

    boss!(problem; model_fitter, acq_maximizer, term_cond, options)
end
