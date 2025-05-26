
"""
    construct_acquisition(::BossProblem) -> (x -> ::Real)
    construct_acquisition(::BossProblem, ::BossOptions) -> (x -> ::Real)

Construct the acquisition function.
"""
construct_acquisition(problem::BossProblem, options::BossOptions=BossOptions()) =
    construct_acquisition(problem.acquisition, problem, options)
