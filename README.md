# BOSS (Bayesian Optimization with Semiparametric Surrogate)

BOSS.jl is a Julia package for Bayesian optimization. It provides a compact way to define an optimization problem and a surrogate model, and solve the problem. It allows changing the algorithms used for the subtasks of fitting the surrogate model and optimizing the acquisition function. Simple interfaces are defined for the use of custom surrogate models and/or algorithms for the subtasks. (See [1] for more information about Bayesian optimization.)

See the [documentation](https://soldasim.github.io/BOSS.jl/) for more information.

## Problem Definition

The problem is defined as follows:

There is some (possibly noisy) blackbox function `y = f(x) = f_true(x) + ϵ` where `ϵ ~ Normal`.

We have some surrogate model `y = model(x) ≈ f_true(x)` describing our limited knowledge about the blackbox function.

We wish to find `x ∈ domain` such that `fitness(f(x))` is maximized while satisfying the constraints `f(x) < cons`.

## The Model

BOSS can be used with purely parametric models (via the `BOSS.Parametric` type), Gaussian Processes (via the `BOSS.Nonparametric` type) or with a semiparametric model (via the `BOSS.Semiparametric`) which combines the two previously mentioned models by supplying the parametric model as the mean of the GP.

## Algorithms

BOSS offers both MAP estimation of model parameters and Bayesian inference (BI) via sampling. 

Currently, the Optimization.jl library is supported for the MAP estimation and the Turing.jl library is supported for the BI sampling. The Optimization.jl library is supported for the acquisition function maximization.

BOSS also provides a simple interface for the use of other custom alagorithms/libraries for model-fitting and/or acquisition maximization by extending the abstract types `BOSS.ModelFitter` and `BOSS.AcquisitionMaximizer`.

## Plotting

A simple plotting script is provided to visualize the optimization process using the Plots.jl package. Use the `PlotCallback` to utilize this feature.

## Examples

See the [documentation](https://soldasim.github.io/BOSS.jl/dev/example/) for example usage.

## References

[1] Bobak Shahriari et al. “Taking the human out of the loop: A review of Bayesian
optimization”. In: Proceedings of the IEEE 104.1 (2015), pp. 148–175

## Citation

If you use this software, please cite it using provided `CITATION.cff` file.
