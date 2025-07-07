
# Functions

This page contains the documentation for all exported functions.

## The Main Function

The main function `bo!(problem::BossProblem; kwargs...)` performs the Bayesian optimization. It augments the dataset (`problem.data`) and updates the model parameters and/or hyperparameters (`problem.params`) of the provided `BossProblem`.

The following diagram showcases the pipeline of the main function `bo!`.

| | | | | |
| --- | --- | --- | --- | --- |
| | | ![BOSS Pipeline](img/boss_pipeline.drawio.png) | | |
| | | | | |

```@docs
bo!
```

The package can be used in two modes;

The "BO mode" is used if the objective function is defined within the [`BossProblem`](@ref). In this mode, BOSS performs the standard Bayesian optimization procedure while querying the objective function for new points.

The "Recommender mode" is used if the objective function is `missing`. In this mode, BOSS performs a single iteration of the Bayesian optimization procedure and returns a recommendation for the next evaluation point. The user can evaluate the objective function manually, use the method [`augment_dataset!`](@ref) to add the result to the data, and call `bo!` again for a new recommendation.

## Utility Functions

```@docs
estimate_parameters!
maximize_acquisition
eval_objective!
update_parameters!
augment_dataset!
construct_acquisition
model_posterior
model_posterior_slice
x_dim
y_dim
cons_dim
data_count
is_consistent
get_fitness
get_params
result
calc_inverse_gamma
TruncatedMvNormal
```
