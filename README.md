[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://soldasim.github.io/BOSS.jl/stable/)

# BOSS (Bayesian Optimization with Semiparametric Surrogate)

BOSS stands for "Bayesian Optimization with Semiparametric Surrogate". BOSS.jl is a Julia package for Bayesian optimization. It provides a straight-forward way to define a BO problem, a surrogate model, and an acquisition function. It allows changing the algorithms used for the subtasks of estimating the model parameters and optimizing the acquisition function. Simple interfaces are defined for the use of custom surrogate models, acquisition functions, and algorithms for the subtasks. Therefore, the package is easily extendable and can be used as a practical skeleton for implementing other BO approaches.

See the [documentation](https://soldasim.github.io/BOSS.jl/) for more information about BOSS.jl.

See [1] for more information about Bayesian optimization.

## Problem Definition

The problem is defined as follows:

There is some (possibly noisy) blackbox function `y = f(x) = f_true(x) + ϵ` where `ϵ ~ Normal`.

We have some surrogate model `y = model(x) ≈ f_true(x)` describing our limited knowledge about the blackbox function.

We wish to find `x ∈ domain` such that `fitness(f(x))` is maximized while satisfying the constraints `f(x) < y_max`.

## The Surrogate Model

BOSS can be used with purely parametric models, Gaussian Processes, or with a semiparametric models combining the two previous models by supplying the parametric model as the mean of the GP. Alternatively, any custom surrogate model can be defined by extending the `SurrogateModel` type.

## Algorithms

BOSS allows defining custom algorithms for the substeps of model parameter estimation and acquisition function maximization. Both MAP estimation of model parameters and Bayesian inference (BI) via sampling are supported. 

Use the `OptimizationMAP` model fitter for MAP estimation via the Optimization.jl library.
Use the `TuringBI` model fitter for BI sampling via the Turing.jl library.
See other available `ModelFitter`s in the [documentation](https://soldasim.github.io/BOSS.jl/).

Use the `OptimizationAM` for acquisition maximization via the Optimization.jl library.
See other available `AcquisitionMaximizer`s in the [documentation](https://soldasim.github.io/BOSS.jl/).

BOSS also provides a simple interface for the use of other custom alagorithms/libraries for model parameter estimation and/or acquisition maximization by extending the abstract types `ModelFitter` and `AcquisitionMaximizer`.

## Examples

See the [documentation](https://soldasim.github.io/BOSS.jl/dev/example/) for example usage.

## Plotting

A simple plotting script is provided to visualize the optimization process using the Plots.jl package. Use the `PlotCallback` to utilize this feature. Only problems with one-dimensional input domains are supported for plotting.

## References

[1] Bobak Shahriari et al. “Taking the human out of the loop: A review of Bayesian
optimization”. In: Proceedings of the IEEE 104.1 (2015), pp. 148–175

## Citation

If you use this software, please cite it using provided `CITATION.cff` file.
