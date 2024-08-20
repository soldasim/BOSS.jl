
"""
    RandomMAP()

Returns random model parameters sampled from their respective priors.

Can be useful with `RandomSelectAM` to avoid unnecessary model parameter estimations.
"""
struct RandomMAP <: ModelFitter{MAP} end

function estimate_parameters(::RandomMAP, problem::BossProblem, options::BossOptions)
    return sample_params(problem.model, problem.noise_std_priors), nothing
end
