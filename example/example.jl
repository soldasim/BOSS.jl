using Random
using Distributions

include("../boss.jl")
include("../utils.jl")

# Load model
# include("model-expcos.jl")
include("model-lincos.jl")
# include("model-exp.jl")
# include("model-sq.jl")

Random.seed!(5555)

# Return symbols used as model parameters.
function f_true(x; noise=0.)
    y = exp(x[1]/10) * cos(2*x[1])
    if noise > 0.
        y += rand(Distributions.Normal(0., noise))
    end
    return y
end

# EXAMPLE - - - - - - - -

function example()
    # experiment settings
    noise = 0.01
    domain_lb = [0.]
    domain_ub = [20.]

    # init data
    init_data_size = 6
    X = hcat(rand(Distributions.Uniform(domain_lb[1], domain_ub[1]), init_data_size))
    Y = [f_true(x; noise) for x in eachrow(X)]

    # test data
    test_data_size = 2000
    X_test = LinRange(domain_lb, domain_ub, test_data_size)
    Y_test = [f_true(x) for x in X_test]

    # run BOSS
    sample_count = 200
    util_opt_multistart = 200

    f_noisy(x) = f_true(x; noise)
    model = get_model()
    time = @elapsed X, Y, plots, errs = optim!(f_noisy, X, Y, model, domain_lb, domain_ub;
        sample_count,
        util_opt_multistart,
        INFO=true,
        max_iters=0,
        X_test,
        Y_test,
        target_err=nothing,
    )

    return time, X, Y, errs, plots
end
