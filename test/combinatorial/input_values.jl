module InputValues

export get_input_vals

import ..PARALLEL_TESTS
using ..BOSS
using ..FileUtils

using Distributions
using KernelFunctions
using LinearAlgebra


# - - - List of All Inputs - - - - -

"""
The names of all parametrized BOSS inputs used in combinatorial testing.
"""
const INPUT_NAMES = load_var_names()


# - - - Objective Function - - - - -

function OBJ_FUNC(x; noise_std=0.1)
    y = exp(x[1]/10) * cos(2*x[1])
    z = (1/2)^6 * (x[1]^2 - (15.)^2)
    
    y += rand(Normal(0., noise_std))
    z += rand(Normal(0., noise_std))

    return [y,z]
end


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

const noise_std_priors_DICT = Dict(
    "with_Dirac" => fill(Dirac(0.1), 2),
    "wo_Dirac" => fill(LogNormal(-2.3, 1.), 2),
    "*" => fill(Dirac(0.1), 2),
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
    "EI" => ExpectedImprovement(),
    "*" => ExpectedImprovement(),
)

const Parametric_predict_DICT = Dict(
    "valid_function" => (x, θ) -> [θ[1] * x[1] * cos(θ[2] * x[1]) + θ[3], 0.],
    "INACTIVE" => nothing,
    "*" => (x, θ) -> [θ[1] * x[1] * cos(θ[2] * x[1]) + θ[3], 0.],
)

const Parametric_theta_priors_DICT = Dict(
    "with_Dirac" => [Normal(0., 1.), Normal(0., 1.), Dirac(0.)],
    "wo_Dirac" => fill(Normal(0., 1.), 3),
    "INACTIVE" => nothing,
    "*" => fill(Normal(0., 1.), 3),
)

const Nonparametric_mean_DICT = Dict(
    "valid_function" => (x) -> [cos(2*x[1]), 0.],
    "nothing" => nothing,
    "INACTIVE" => nothing,
    "*" => nothing,
)

const Nonparametric_kernel_DICT = Dict(
    "valid" => Matern32Kernel(),
    "INACTIVE" => nothing,
    "*" => Matern32Kernel(),
)

const Nonparametric_amplitude_priors_DICT = Dict(
    "with_Dirac" => fill(Dirac(1.), 2),
    "wo_Dirac" => fill(LogNormal(), 2),
    "INACTIVE" => nothing,
    "*" => fill(LogNormal(), 2),
)

const Nonparametric_lengthscale_priors_DICT = Dict(
    "with_Dirac" => fill(product_distribution([Dirac(1.)]), 2),
    "wo_Dirac" => fill(MvLogNormal(0.1*ones(1), I(1)), 2),
    "INACTIVE" => nothing,
    "*" => fill(MvLogNormal(0.1*ones(1), I(1)), 2),
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
    "optimization_map" => :Optimization,
    "turing_bi" => :Turing,
    "sampling_map" => :Sampling,
    "random_map" => :Random,            # as default only
    # "sample_opt_map" => :SampleOpt,   # excluded from tests
    "*" => :Random,
)

const AcquisitionMaximizer_DICT = Dict(
    "optimization" => :Optimization,
    "grid" => :Grid,
    "sampling" => :Sampling,
    "random" => :Random,                # as defualt only
    # "sample_opt" => :SampleOpt,       # excluded from tests
    "*" => :Random,
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

end # module InputValues
