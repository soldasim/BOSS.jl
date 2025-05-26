
# BOSS.jl

BOSS stands for "Bayesian Optimization with Semiparametric Surrogate". BOSS.jl is a Julia package for Bayesian optimization. It provides a straight-forward way to define a BO problem, a surrogate model, and an acquisition function. It allows changing the algorithms used for the subtasks of estimating the model parameters and optimizing the acquisition function. Simple interfaces are defined for the use of custom surrogate models, acquisition functions, and algorithms for the subtasks. Therefore, the package is easily extendable and can be used as a practical skeleton for implementing other BO approaches.

## Bayesian Optimization

Bayesian optimization is a general algorithm for blackbox optimization (or active learning) with expensive-to-evaluate objective functions. The main focus is on optimal experiment design (i.e. selection of optimal evaluation points) to minimize the number of required objective function evaluations. The general procedure is showcased by the pseudocode below. [1]

| | | | | |
| --- | --- | --- | --- | --- |
| | | ![Bayesian optimization](img/bo.png) | | |
| | | | | |

The real-valued acquisition function ``\alpha(x)`` defines the utility of a potential point for the next evaluation. We maximize the acquisition function to select the most useful point for the next evaluation.

Once the optimum ``x_{n+1}`` of the acquisition function is obtained, we evaluate the blackbox objective function ``y_{n+1} \gets f(x_{n+1})`` and we augment the dataset.

Finally, we update the surrogate model according to the augmented dataset.

See [1] for more information on Bayesian optimization.

## Optimization Problem

The problem is defined using the [`BossProblem`](@ref) structure and it follows the formalization below.

We have some (noisy) blackbox objective function
```math
y = f(x) = f_t(x) + \epsilon \;,
```
where ``\epsilon \sim \mathcal{N}(0, \sigma_f^2)`` is a Gaussian noise. We are able to evaluate ``f(x_i)`` and obtain a noisy realization
```math
y_i \sim \mathcal{N}(f_t(x_i), \sigma_f^2) \;.
```

Our goal is to solve the following optimization problem
```math
\begin{aligned}
\text{max} \; & \text{fit}(y) \\
\text{s.t.} \; & y < y_\text{max} \\
& x \in \text{Domain} \;,
\end{aligned}
```
where ``\text{fit}(y)`` is a real-valued fitness function defined, ``y_\text{max}`` is a vector defining constraints on outputs, and ``\text{Domain}`` defines constraints on inputs.

## Surrogate Model

The surrogate model approximates the objective function based on the available data. It is defined using the [`SurrogateModel`](@ref) type and passed to the [`BossProblem`](@ref) structure. The basic provided models are the [`Parametric`](@ref) model, the [`GaussianProcess`](@ref), and the [`Semiparametric`](@ref) model combining the previous two.

The predictive distribution of the [`Parametric`](@ref) model
```math
y \sim \mathcal{N}(m(x; \hat\theta), \hat\sigma_f^2)
```
is given by the parametric function ``m(x; \theta)``, the estimated parameter vector ``\hat\theta``, and the estimated evaluation noise deviations ``\hat\sigma_f``. The model is defined by the parametric function ``m(x; \theta)`` together with parameter priors ``\theta_i \sim p(\theta_i)``. The parameters ``\hat\theta`` and the noise deviations ``\hat\sigma_f`` are estimated using the `ModelFitter` based on the current dataset.

The [`GaussianProcess`](@ref) (GP) is a nonparametric model, so its predictive distribution is based on the whole dataset instead of some vector of parameters. The predictive distribution is given by equations 29, 30 in [1]. The model is defined by choosing priors for all its hyperparameters (length scales, amplitudes, and noise deviation).

The [`Semiparametric`](@ref) model combines the previous two models. It is a Gaussian process, but uses the parametric model as the prior mean function of the GP (the ``\mu_0(x)`` function in equation 29 in [1]). An alternative way of interpreting the semiparametric model is that it fits the data using a parametric model and uses a Gaussian process to model the residual errors of the parametric model. The model is defined by defining both a [`Parametric`](@ref) model and a [`GaussianProcess`](@ref) and passing them to the [`Semiparametric`](@ref) model.

A custom surrogate model can be defined by subtyping the [`SurrogateModel`](@ref) type and implementing the defined API.

## Acquisition Function

The acquisition function is defined using subtypes of the [`AcquisitionFunction`](@ref) type and passed to the [`BossProblem`](@ref). In case of optimization problems, the fitness is defined as a part of the [`AcquisitionFunction`](@ref).

The most commonly used acquisition function, [`ExpectedImprovement`](@ref), is provided. Custom acquisition functions can be defined by extending the [`AcquisitionFunction`](@ref) type.

## The Sub-Algorithms

The algorithms which are used to perform the sub-steps of the BO procedure are defined using the subtypes of the [`ModelFitter`](@ref) type and the [`AcquisitionMaximizer`](@ref) type. They are passed to the main function [`bo!`](@ref) as keyword arguments.

The model parameters can either be estimated in a MAP fashion (using for example the [`OptimizationMAP`](@ref) model fitter) or sampled via Bayesian inference (using the [`TuringBI`](@ref) model fitter). Other model fitters are also available and custom ones can be defined by extending the [`ModelFitter`](@ref) type.

The most basic [`AcquisitionMaximizer`](@ref) is the [`OptimizationAM`](@ref). Other acquisition maximizers are also available and custom ones can be defined by extending the [`AcquisitionMaximizer`](@ref) type. Batching of the objective function evaluations can be achieved by wrapping any other acquisition maximizer in [`SequentialBatchAM`](@ref).

## Termination Condition

The termination condition of the BO procedure can be defined using the [`TermCond`](@ref) type. The termination condition is passed to the main function [`bo!`](@ref) as a keyword argument.

The basic [`IterLimit`](@ref) termination condition is provided. Custom termination conditions can be defined by extending the [`TermCond`](@ref) type.

## Active Learning Problem

The BOSS.jl package currently only supports optimization problems out-of-the-box. However, BOSS.jl can be easily adapted for active learning by defining a suitable acquisition function (such as information gain or Kullback-Leibler divergence) to use instead of the expected improvement (see [`AcquisitionFunction`](@ref)).

## References

[1] Bobak Shahriari et al. “Taking the human out of the loop: A review of Bayesian
optimization”. In: Proceedings of the IEEE 104.1 (2015), pp. 148–175
