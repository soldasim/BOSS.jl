
"""
    RandomFitter()

Returns random model parameters sampled from their respective priors.

Can be useful with `RandomSelectAM` to avoid unnecessary model parameter estimations.
"""
struct RandomFitter <: ModelFitter{RandomParams} end

function estimate_parameters(::RandomFitter, problem::BossProblem, options::BossOptions)
    return RandomParams(params_sampler(problem.model, problem.data)())
end
