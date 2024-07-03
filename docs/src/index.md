
# Documentation

## Main Function

```@docs
bo!
```

## Problem Definition

```@docs
BossProblem
```

### Fitness

```@docs
Fitness
NoFitness
LinFitness
NonlinFitness
```

### Domain

```@docs
Domain
```

### Constraints on Output

`y_max`

### Surrogate Model

```@docs
SurrogateModel
Parametric
LinModel
NonlinModel
Nonparametric
GaussianProcess
Semiparametric
```

### Evaluation Noise

`noise_std_priors`

### Experiment Data

```@docs
ExperimentData
ExperimentDataPrior
ExperimentDataPost
ExperimentDataMLE
ExperimentDataBI
```

## Model Fitter

```@docs
ModelFitter
ModelFit
OptimizationMLE
TuringBI
SamplingMLE
RandomMLE
```

## Acquisition Maximizer

```@docs
AcquisitionMaximizer
OptimizationAM
GridAM
RandomAM
SequentialBatchAM
```

## Acquisition Function

```@docs
AcquisitionFunction
ExpectedImprovement
```

## Termination Conditions

```@docs
TermCond
IterLimit
```

## Miscellaneous

```@docs
result
BossOptions
BossCallback
NoCallback
PlotCallback
```
