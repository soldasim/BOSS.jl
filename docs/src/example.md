
# Example

An illustrative problem is solved using BOSS.jl here. The source code is available at [github](https://github.com/soldasim/BOSS.jl/blob/master/scripts/example.jl).

## The Problem

We have an expensive-to-evaluate blackbox function `blackbox(x) -> (y, z)`.

```julia
function blackbox(x; noise_std=0.1)
    y = exp(x[1]/10) * cos(2*x[1])
    z = (1/2)^6 * (x[1]^2 - (15.)^2)
    
    y += rand(Normal(0., noise_std))
    z += rand(Normal(0., noise_std))

    return [y,z]
end
```

We wish to maximize `y` such that `x ∈ <0, 20>` (constraint on input) and `z < 0` (constraint on output).

## Problem Definition

First, we define the problem as an instance of [`BossProblem`](@ref).

```julia
problem() = BossProblem(;
    f = blackbox,
    domain = Domain(;
        bounds = ([0.], [20.]),     # x ∈ <0, 20>
    ),
    y_max = [Inf, 0.],              # z < 0
    acquisition = ExpectedImprovement(;
        fitness = LinFitness([1, 0]),   # maximize y
    ),
    model = nonparametric(), # or `parametric()` or `semiparametric()`
    data = init_data(),
)
```

We use [`Fitness`](@ref) to define the objective. Here, the `LinFitness([1, 0])` specifies that we wish to maximize `1*y + 0*z`. (See also [`NonlinFitness`](@ref).)

We use the keyword `f` to provide the blackbox objective function.

We use the [`Domain`](@ref) structure to define the constraints on inputs.

We use the keyword `y_max` to define the constraints on outputs.

## Surrogate Model Definition

Now we define the surrogate model used to approximate the objective function based on the available data from previous evaluations.

### Gaussian Process

Usually, we will use a Gaussian process.

```julia
nonparametric() = GaussianProcess(;
    kernel = BOSS.Matern32Kernel(),
    amplitude_priors = amplitude_priors(),
    lengthscale_priors = lengthscale_priors(),
    noise_std_priors = noise_std_priors(),
)
```

### Parametric Model

If we have some knowledge about the blackbox function, we can define a parametric model.

```julia
parametric() = NonlinearModel(;
    predict = (x, θ) -> [
        θ[1] * x[1] * cos(θ[2] * x[1]) + θ[3],
        0.,
    ],
    theta_priors = fill(Normal(0., 1.), 3),
    noise_std_priors = noise_std_priors(),
)
```

The function `predict(x, θ) -> y` defines our parametric model where `θ` are the model parameters which will be fitted based on the data.

The keyword `theta_priors` is used to define priors on the model parameters `θ`. The priors can be used to include our expert knowledge, to regularize the model, or a uniform prior can be used to not bias the model fit.

### Semiparametric Model

We can use the parametric model together with a Gaussian process to define the semiparametric model.

```julia
semiparametric() = Semiparametric(
    parametric(),
    nonparametric(),
)
```

This allows us to leverage our expert knowledge incorporated in the parametric model while benefiting from the flexibility of the Gaussian process.

## Hyperparameter Priors

We need to define all hyperparameters. Instead of defining scalar values, we will define priors over them and let BOSS fit their values based on the data. This alleviates the importance of our choice and allows for Bayesian inference if we wish to use it.

(If one wants to define some hyperparameters as scalars instead, a `Dirac` prior can be used and the hyperparameters will be skipped from model fitting.)

### Evaluation Noise

BOSS assumes Gaussian evaluation noise on the objective blackbox function. Noise std priors define our belief about the standard deviation of the noise of each individual output dimension.

```julia
noise_std_priors() = fill(truncated(Normal(0., 0.1); lower=0.), 2)
# noise_std_priors() = fill(Dirac(0.1), 2)
```

### Amplitude

The amplitude of the Gaussian process expresses the expected deviation of the output values. We again define an amplitude prior for each individual output dimension.

```julia
amplitude_priors() = fill(truncated(Normal(0., 5.); lower=0.), 2)
# amplitude_priors() = fill(Dirac(5.), 2)
```

### Length Scales

Informally, the length scales of the Gaussian process define how far within the input domain does the model extrapolate the information obtained from the dataset. For each output dimension, we define a multivariate prior over all input dimensions. (In our case two 1-variate priors.)

```julia
lengthscale_priors() = fill(Product([truncated(Normal(0., 20/3); lower=0.)]), 2)
# lengthscale_priors() = fill(Product(fill(Dirac(1.), 1)), 2)
```

## Model Fitter

We can specify the algorithm used to fit the model hyperparameters using the [`ModelFitter`](@ref) type.

We can fit the hyperparameters in a MAP fashion using the [`OptimizationMAP`](@ref) model fitter together with any algorithm from `Optimization.jl` and its extensions.

```julia
using OptimizationPRIMA

map_fitter() = OptimizationMAP(;
    algorithm = NEWUOA(),
    multistart = 20,
    parallel = false,
    rhoend = 1e-4,
)
```

Or we can use Bayesian inference and sample the parameters from their posterior (given by the priors and the data likelihood) using the [`TuringBI`](@ref) model fitter.

```julia
using Turing

bi_fitter() = TuringBI(;
    sampler = NUTS(1000, 0.65),
    warmup = 100,
    samples_in_chain = 10,
    chain_count = 8,
    leap_size = 5,
    parallel = true,
)
```

See also [`SamplingMAP`](@ref) and [`RandomFitter`](@ref) for more trivial model fitters suitable for simple experimentation with the package.

## Acquisition Maximizer

We can specify the algorithm used to maximize the acquisition function (in order to select the next evaluation point) by using the [`AcquisitionMaximizer`](@ref) type.

We can use the [`OptimizationAM`](@ref) maximizer together with any algorithm from `Optimization.jl`.

```julia
acq_maximizer() = OptimizationAM(;
    algorithm = BOBYQA(),
    multistart = 20,
    parallel = false,
    rhoend = 1e-4,
)
```

Make sure to use an algorithm suitable for the given domain. (For example, in our case the domain is bounded by box constraints only, so we the BOBYQA optimization algorithm designed for box constraints problems.)

See also [`GridAM`](@ref), [`RandomAM`](@ref) for more trivial acquisition maximizers suitable for simple experimentation with the package.

The [`SequentialBatchAM`](@ref) can be used to wrap any of the other acquisition maximizers to enable objective function evaluation in batches.

## Acquisition Function

The acquisition function defines how the next evaluation point is selected in each iteration. The acquisition function is maximized by the acquisition maximizer algorithm (discussed in the previous section).

Currently, the only implemented acquisition function is the [`ExpectedImprovement`](@ref) acquisition most commonly used in Bayesian optimization.

## Miscellaneous

Finally, we can define the termination condition using the [`TermCond`](@ref) type. Currently, the only available termination condition is the trivial [`IterLimit`](@ref) condition. (However, one can simply define his own termination condition by extending the [`TermCond`](@ref) type.)

The [`BossOptions`](@ref) structure can be used to change other miscellaneous settings.

The [`PlotCallback`](@ref) can be provided in [`BossOptions`](@ref) to enable plotting of the BO procedure. This can be useful for initial experimentation with the package. Note that the plotting only works for 1-dimensional input domains.

```julia
using Plots

options() = BossOptions(;
    info = true,
    callback = PlotCallback(Plots;
        f_true = x->blackbox(x; noise_std=0.),
    ),
)
```

## Initial Data

We have to provide at least a single initial data point.

```julia
function init_data()
    X = [10.;;]
    Y = hcat(blackbox.(eachcol(X))...)
    return ExperimentData(X, Y)
end
```

## Run BOSS

Once we define the problem and all hyperparameters, we can run the BO procedure by calling the `bo!` function.

```julia
prob = bo!(problem();
    model_fitter = map_fitter(), # or `bi_fitter()`
    acq_maximizer = acq_maximizer(),
    term_cond = IterLimit(10),
    options = options(),
);
```
