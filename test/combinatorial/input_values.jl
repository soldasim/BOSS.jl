
function OBJ_FUNC(x; noise_var=0.01)
    y = exp(x[1]/10) * cos(2*x[1])
    z = (1/2)^6 * (x[1]^2 - (15.)^2)
    
    y += rand(BOSS.Normal(0., sqrt(noise_var)))
    z += rand(BOSS.Normal(0., sqrt(noise_var)))

    return [y,z]
end

"""
The names of all parametrized BOSS inputs used in combinatorial testing.
"""
const INPUT_NAMES = [
    "XY",
    "f",
    "bounds",
    "discrete",
    "cons",
    "y_max",
    "noise_var_priors",
    "ModelFitter_parallel",
    "AcquisitionMaximizer_parallel",
    "Acquisition",
    "Parametric_predict",
    "Parametric_theta_priors",
    "Nonparametric_mean",
    "Nonparametric_kernel",
    "Nonparametric_length_scale_priors",
    "Semiparametric_mean",
    "MODEL",
    "LinFitness_coefs",
    "NonlinFitness_fit",
    "FITNESS",
    "iter_max",
    "ModelFitter",
    "AcquisitionMaximizer",
    "VALID",
]


# - - - ValueName-to-Value Dicitionaries - - - - -

function _XY(X)
    Y = reduce(hcat, OBJ_FUNC.(eachcol(X)))
    return X, Y
end
const XY_DICT = Dict(
    "duplicates" => _XY([5.;; 10.;; 10.;;]),
    "noduplicates" => _XY([5.;; 10.;; 15.;;]),
    "*" => _XY([5.;; 10.;; 15.;;]),
)

const f_DICT = Dict(
    "valid_function" => OBJ_FUNC,
    "missing" => missing,
    "*" => OBJ_FUNC,
)

const bounds_DICT = Dict(
    "all_finite" => ([0.], [20.]),
    "*" => ([0.], [20.]),
)

const discrete_DICT = Dict(
    "some_true" => [true],
    "all_false" => [false],
    "*" => [false],
)

const cons_DICT = Dict(
    "valid_function" => (x) -> [x[1] - 5.],
    "nothing" => nothing,
    "*" => nothing,
)

const y_max_DICT = Dict(
    "some_finite" => [Inf, 0.],
    "all_infinite" => [Inf, Inf],
    "*" => [Inf, 0.],
)

const noise_var_priors_DICT = Dict(
    "with_Dirac" => fill(BOSS.Dirac(0.1), 2),
    "wo_Dirac" => fill(BOSS.LogNormal(-2.3, 1.), 2),
    "*" => fill(BOSS.Dirac(0.1), 2),
)

const ModelFitter_parallel_DICT = Dict(
    "true" => PARALLEL_TESTS ? true : false,
    "false" => false,
    "*" => PARALLEL_TESTS ? true : false,
)

const AcquisitionMaximizer_parallel_DICT = Dict(
    "true" => PARALLEL_TESTS ? true : false,
    "false" => false,
    "*" => PARALLEL_TESTS ? true : false,
)

const Acquisition_DICT = Dict(
    "EI" => BOSS.ExpectedImprovement(),
    "*" => BOSS.ExpectedImprovement(),
)

const Parametric_predict_DICT = Dict(
    "valid_function" => (x, θ) -> [θ[1] * x[1] * cos(θ[2] * x[1]) + θ[3], 0.],
    "INACTIVE" => nothing,
    "*" => (x, θ) -> [θ[1] * x[1] * cos(θ[2] * x[1]) + θ[3], 0.],
)

const Parametric_theta_priors_DICT = Dict(
    "with_Dirac" => [BOSS.Normal(0., 1.), BOSS.Normal(0., 1.), BOSS.Dirac(0.)],
    "wo_Dirac" => fill(BOSS.Normal(0., 1.), 3),
    "INACTIVE" => nothing,
    "*" => fill(BOSS.Normal(0., 1.), 3),
)

const Nonparametric_mean_DICT = Dict(
    "valid_function" => (x) -> [cos(2*x[1]), 0.],
    "nothing" => nothing,
    "INACTIVE" => nothing,
    "*" => nothing,
)

const Nonparametric_kernel_DICT = Dict(
    "valid" => BOSS.Matern52Kernel(),
    "INACTIVE" => nothing,
    "*" => BOSS.Matern52Kernel(),
)

const Nonparametric_length_scale_priors_DICT = Dict(
    "with_Dirac" => fill(BOSS.product_distribution([BOSS.Dirac(1.)]), 2),
    "wo_Dirac" => fill(BOSS.MvLogNormal(0.1*ones(1), 1.0*ones(1)), 2),
    "INACTIVE" => nothing,
    "*" => fill(BOSS.MvLogNormal(0.1*ones(1), 1.0*ones(1)), 2),
)

const Semiparametric_mean_DICT = Dict(
    "nothing" => nothing,
    "INACTIVE" => nothing,
    "*" => nothing,
)

const MODEL_DICT = Dict(
    "Parametric" => :Parametric,
    "Nonparametric" => :Nonparametric,
    "Semiparametric" => :Semiparametric,
    "*" => :Semiparametric,
)

const LinFitness_coefs_DICT = Dict(
    "wo_infs" => [1., 0.],
    "INACTIVE" => nothing,
    "*" => [1., 0.],
)

const NonlinFitness_fit_DICT = Dict(
    "valid_function" => (y) -> y[1],
    "INACTIVE" => nothing,
    "*" => (y) -> y[1],
)

const FITNESS_DICT = Dict(
    "LinFitness" => :LinFitness,
    "NonlinFitness" => :NonlinFitness,
    "*" => :LinFitness,
)

const iter_max_DICT = Dict(
    "1" => 1,
    "2" => 2,
    "*" => 1,
)

const ModelFitter_DICT = Dict(
    "optimization_mle" => :Optimization,
    "turing_bi" => :Turing,
    "sampling_mle" => :Sampling,
    "random_mle" => :Random,
    "*" => :Optimization,
)

const AcquisitionMaximizer_DICT = Dict(
    "optimization" => :Optimization,
    "grid" => :Grid,
    "random" => :Random,
    "*" => :Optimization,
)

const VALID_DICT = Dict(
    "true" => true,
    "false" => false,
)


# - - - VariableName-to-ValueDict Dictionary - - - - -

macro inputdict(expr)
    names = eval(expr)
    pairs = (:($(name) => $(Symbol(name*"_DICT"))) for name in names)
    return Expr(:call, :Dict, pairs...)
end

const INPUT_DICT = @inputdict INPUT_NAMES

"""
Gets a combination from the csv file and returns a function mapping
the input variable names to concrete values.

# In
`comb`: A dictionary mapping input variable names to input value names.

# Out
`var -> val`: A function mapping input variable names to input values.
"""
function get_input_vals(comb)
    return var -> INPUT_DICT[var][comb[var]]
end
