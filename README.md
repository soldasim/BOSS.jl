# BOSS (Bayesian Optimization with Semiparametric Surrogate)

BOSS is a julia package for Bayesian optimization. It provides a compact way to define an optimization problem and a surrogate model and solve the problem. It allows to change the hyperparameters of the underlying algorithms and provides a simple interface to use custom algorithms for the subtasks as well.

## Problem Definition

The problem is defined as follows:

There is some (possibly noisy) blackbox function `y = f(x) = f_true(x) + ϵ` where `ϵ ~ Normal`.

We have some surrogate model `y = model(x) ≈ f_true(x)` describing our limited knowledge about the blackbox function.

We wish to find `x ∈ domain` such that `fitness(f(x))` is maximized while satisfying the constraints `f(x) < cons`.

## The Model

BOSS can be used with purely parametric models (via the `BOSS.Parametric` type), Gaussian Processes (via the `BOSS.Nonparametric` type) or with a semiparametric model (via the `BOSS.Semiparametric`) which combining the two previously mentioned models by supplying the parametric model as the mean of the GP.

## Algorithms

BOSS offers both MLE estimation of model parameters and Bayesian inference via sampling. 

Currently, the Optim.jl library is supported for the MLE estimation and the Turing.jl library is supported for the BI sampling. The Optim.jl library is supported for the acquisition function maximization. Support for other libraries will be added in the future.

BOSS also provides a simple interface for the use of custom alagorithms/libraries for model-fitting and/or acquisition maximization by inheriting the `BOSS.ModelFitter` and `BOSS.AcquisitionMaximizer` types.

## Plots

BOSS provides a simple plotting script to visualize the optimization process using the Plots.jl package. To use this feature pass the `Plots` module via the `BOSS.PlotOptions` structure to the BOSS algorithm.

## Examples

See https://github.com/Sheld5/BOSS.jl/tree/master/scripts for example usage.
