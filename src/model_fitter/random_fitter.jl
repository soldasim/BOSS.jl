
"""
    RandomMLE()

Returns random model parameters sampled from their respective priors.

Can be useful with `RandomSelectAM` to avoid unnecessary model parameter estimations.
"""
struct RandomMLE <: ModelFitter{MLE} end

function estimate_parameters(::RandomMLE, problem::OptimizationProblem, options::BossOptions)
    return sample_params(problem.model, problem.noise_var_priors)
end
