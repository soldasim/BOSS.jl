
# Functions

This page contains the documentation for all exported functions.

## Main Function

The main function `bo!(::BossProblem; kwargs...)` performs the Bayesian optimization. It augments the dataset and updates the model parameters and/or hyperparameters stored in `problem.data`.

```@docs
bo!
```

The following diagram showcases the pipeline of the main function. The package can be used in two modes;

The "BO mode" is used if the objective function is defined within the [`BossProblem`](@ref). In this mode, BOSS performs the standard Bayesian optimization procedure while querying the objective function for new points.

The "Recommender mode" is used if the objective function is `missing`. In this mode, BOSS performs a single iteration of the Bayesian optimization procedure and returns a recommendation for the next evaluation point. The user can evaluate the objective function manually, use the method [`augment_dataset!`](@ref) to add the result to the data, and call BOSS again for a new recommendation.

| | | | | |
| --- | --- | --- | --- | --- |
| | | ![BOSS Pipeline](img/boss_pipeline.drawio.png) | | |
| | | | | |

## Utility Functions

```@docs
estimate_parameters!
maximize_acquisition
eval_objective!
augment_dataset!
model_posterior
model_posterior_slice
average_posterior
result
```
