
"""
    construct_acquisition(::BossProblem) -> (x -> ::Real)
    construct_acquisition(::BossProblem, ::BossOptions) -> (x -> ::Real)

Construct the acquisition function.
"""
construct_acquisition(problem::BossProblem, options::BossOptions=BossOptions()) =
    construct_acquisition(problem.acquisition, problem, options)

"""
    construct_safe_acquisition(::BossProblem) -> (x -> ::Real)
    construct_safe_acquisition(::BossProblem, ::BossOptions) -> (x -> ::Real)

Construct a "safe" version of the acquisition function.
The safe version return `-Inf` instead of throwing an error.
"""
construct_safe_acquisition(problem::BossProblem, options::BossOptions=BossOptions()) =
    construct_safe_acquisition(problem.acquisition, problem, options)

function construct_safe_acquisition(acquisition::AcquisitionFunction, problem::BossProblem, options::BossOptions)
    acq = construct_acquisition(acquisition, problem, options)
    acq_safe = make_safe(acq, -Inf; options.info, options.debug)
    return acq_safe
end
