# Data Types & Structures

The following diagram showcases the type hierarchy of all BOSS inputs and hyperparameters.

| | | | | |
| --- | --- | --- | --- | --- |
| | | ![BOSS Pipeline](img/boss_inputs.drawio.png) | | |
| | | | | |

This reminder of this page contains documentation for all exported types and structures.

## Problem Definition

The `BossProblem` structure contains the whole problem definition, the model definition, and the data together with the current parameter and hyperparameter values.

```@docs
BossProblem
```

## Fitness

The `Fitness` type is used to define the fitness function ``\text{fit}(y) \rightarrow \mathbb{R}``.

The `NoFitness` can be used in problems without defined fitness (such as active learning problems). It is the default option used if no fitness is provided to [`BossProblem`](@ref). The `NoFitness` can only be used with [`AcquisitionFunction`](@ref) that does not require fitness.

The `LinFitness` can be used to define a simple linear fitness function
```math
\text{fit}(y) = \alpha^T y \;.
```
Using `LinFitness` instead of `NonlinFitness` may allow for simpler/faster computation of some acquisition functions.

The `NonlinFitness` can be used to define an arbitrary fitness function
```math
\text{fit}(y) \rightarrow \mathbb{R} \;.
```

```@docs
Fitness
NoFitness
LinFitness
NonlinFitness
```

## Input Domain

The `Domain` structure is used to define the input domain ``x \in \text{Domain}``. The domain is formalized as
```math
\begin{aligned}
& lb < x < ub \\
& d_i \implies (x_i \in \mathbb{Z}) \\
& \text{cons}(x) > 0 \;.
\end{aligned}
```

```@docs
Domain
AbstractBounds
```

## Output Constraints

Constraints on output vector `y` can be defined using the `y_max` field. Providing `y_max` to [`BossProblem`](@ref) defines the linear constraints `y < y_max`.

Arbitrary nonlinear constraints can be defined by augmenting the objective function. For example to define the constraint `y[1] * y[2] < c`, one can define an augmented objective function
```julia
function f_c(x)
    y = f(x)  # the original objective function
    y_c = [y..., y[1] * y[2]]
    return y_c
end
```
and use
```julia
y_max = [fill(Inf, y_dim)..., c]
```
where `y_dim` is the output dimension of the original objective function. Note that defining nonlinear constraints this way increases the output dimension of the objective function and thus the model definition has to be modified accordingly.

## Surrogate Model

The surrogate model is defined using the [`SurrogateModel`](@ref) type.

```@docs
SurrogateModel
```

The `LinModel` and `NonlinModel` structures are used to define parametric models. (Some compuatations are simpler/faster with linear model, so the `LinModel` might provide better performance in the future. This functionality is not implemented yet, so using the `NonlinModel` is equiavalent for now.)

```@docs
Parametric
LinModel
NonlinModel
```

The `GaussianProcess` structure is used to define a Gaussian process model. See [1] for more information about Gaussian processes.

```@docs
Nonparametric
GaussianProcess
```

The `Semiparametric` structure is used to define a semiparametric model combining the parametric and nonparametric (Gaussian process) models.

```@docs
Semiparametric
```

## Parameters & Hyperparameters

The `BOSS.ModelParams` and `BOSS.ParamPriors` type aliases are used throughout the package to pass around model (hyper)parameters and their priors. These types are only important for advanced usage of BOSS. (E.g. implementing custom surrogate models.)

```@docs
BOSS.ModelParams
BOSS.Theta
BOSS.LengthScales
BOSS.Amplitudes
BOSS.NoiseStd
```

```@docs
BOSS.ParamPriors
BOSS.ThetaPriors
BOSS.LengthScalePriors
BOSS.AmplitudePriors
BOSS.NoiseStdPriors
```

All (hyper)parameter priors are defined as a part of the surrogate model definition. All surrogate models share the `noise_std_priors` field, but other priors may be missing depending on the particular model.

## Experiment Data

The data from all past objective function evaluations as well as estimated parameter and/or hyperparameter values of the surrogate model are stored in the `ExperimentData` types.

```@docs
ExperimentData
```

The `ExperimentDataPriors` structure is used to pass the initial dataset to the [`BossProblem`](@ref).

```@docs
ExperimentDataPrior
```

The `ExperimentDataPost` types contain the estimated model (hyper)parameters in addition to the dataset. The `ExperimentDataMAP` structure contains the MAP estimate of the parameters in case a MAP model fitter is used, and the `ExperimentDataBI` structure contains samples of the parameters in case a Bayesian inference model fitter is used.

```@docs
ExperimentDataPost
ExperimentDataMAP
ExperimentDataBI
```

## Model Fitter

The `ModelFitter` type defines the algorithm used to estimate the model (hyper)parameters.

```@docs
ModelFitter
ModelFit
```

The `OptimizationMAP` model fitter can be used to utilize any optimization algorithm from the Optimization.jl package in order to find the MAP estimate of the (hyper)parameters. (See the example usage.)

```@docs
OptimizationMAP
```

The `TuringBI` model fitter can be used to utilize the Turing.jl library in order to sample the (hyper)parameters from the posterior given by the current dataset.

```@docs
TuringBI
```

The `SamplingMAP` model fitter preforms MAP estimation by sampling the parameters from their priors and maximizing the posterior probability over the samples. This is a trivial model fitter suitable for simple experimentation with BOSS.jl and/or Bayesian optimization. A more sophisticated model fitter such as `OptimizationMAP` or `TuringBI` should be used to solve real problems.

```@docs
SamplingMAP
```

The `RandomMAP` model fitter samples random parameter values from their priors. It does NOT optimize for the most probable parameters in any way. This model fitter is provided solely for easy experimentation with BOSS.jl and should not be used to solve problems.

```@docs
RandomMAP
```

The `SampleOptMAP` model fitter combines the `SamplingMAP` and `OptimizationMAP`. It first samples many model parameter samples from their priors, and subsequently runs multiple optimization runs initiated at the best samples.

```@docs
SampleOptMAP
```

## Acquisition Maximizer

The `AcquisitionMaximizer` type is used to define the algorithm used to maximize the acquisition function.

```@docs
AcquisitionMaximizer
```

The `OptimizationAM` can be used to utilize any optimization algorithm from the Optimization.jl package.

```@docs
OptimizationAM
```

The `GridAM` maximizes the acquisition function by evaluating all points on a fixed grid of points. This is a trivial acquisition maximizer suitable for simple experimentation with BOSS.jl and/or Bayesian optimization. More sophisticated acquisition maximizers such as `OptimizationAM` should be used to solve real problems.

```@docs
GridAM
```

The `SamplingAM` samples random candidate points from the given `x_prior` distribution
and selects the sample with maximal acquisition value.

```@docs
SamplingAM
```

The `RandomAM` simply returns a random point. It does NOT perform any optimization. This acquisition maximizer is provided solely for easy experimentation with BOSS.jl and should not be used to solve problems.

```@docs
RandomAM
```

The `GivenPointAM` always return the same evaluation point predefined by the user. The `GivenSequenceAM` returns the predefined sequence of evaluation points and throws an error once it runs out of points. These dummy acquisition maximizers are useful for controlled experiments.

```@docs
GivenPointAM
GivenSequenceAM
```

The `SampleOptAM` samples many candidate points from the given `x_prior` distribution,
and subsequently performs multiple optimization runs initiated from the best samples.

```@docs
SampleOptAM
```

The `SequentialBatchAM` can be used as a wrapper of any of the other acquisition maximizers. It returns a batch of promising points for future evaluations instead of a single point, and thus allows for evaluation of the objective function in batches.

```@docs
SequentialBatchAM
```

## Acquisition Function

The acquisition function is defined using the `AcquisitionFunction` type.

```@docs
AcquisitionFunction
```

The `ExpectedImprovement` defines the expected improvement acquisition function. See [1] for more information.

```@docs
ExpectedImprovement
```

## Termination Conditions

The `TermCond` type is used to define the termination condition of the BO procedure.

```@docs
TermCond
```

The `NoLimit` can be used to let the algorithm run indefinitely.

```@docs
NoLimit
```

The `IterLimit` terminates the procedure after a predefined number of iterations.

```@docs
IterLimit
```

## Miscellaneous

The `BossOptions` structure is used to define miscellaneous hyperparameters of the BOSS.jl package.

```@docs
BossOptions
```

The `BossCallback` type is used to pass callbacks which will be called in every iteration of the BO procedure (and once before the procedure starts).

```@docs
BossCallback
NoCallback
```

The `PlotCallback` provides plots the state of the BO procedure in every iteration. It currently only supports one-dimensional input spaces.

```@docs
PlotCallback
```

# References

[1] Bobak Shahriari et al. “Taking the human out of the loop: A review of Bayesian
optimization”. In: Proceedings of the IEEE 104.1 (2015), pp. 148–175
