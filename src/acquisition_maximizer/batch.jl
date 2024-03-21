
"""
    SequentialBatchAM(::AcquisitionMaximizer, ::Int)
    SequentialBatchAM(; am, batch_size)

Provides multiple candidates for batched objective function evaluation.

Selects the candidates sequentially by iterating the following steps:
- `) Use the 'inner' acquisition maximizer to select a candidate `x`.
- 2) Extend the dataset with a 'speculative' new data point
    created by taking the candidate `x` and the posterior predictive mean of the surrogate `ŷ`.
- 3) If `batch_size` candidates have been selected, return them.
    Otherwise, goto step 1).

# Keywords
- `am::AcquisitionMaximizer`: The inner acquisition maximizer.
- `batch_size::Int`: The number of candidates to be selected.
"""
struct SequentialBatchAM{
    AM<:AcquisitionMaximizer
} <: AcquisitionMaximizer
    am::AM
    batch_size::Int
end
SequentialBatchAM(;
    am,
    batch_size,
) = SequentialBatchAM(am, batch_size)

function maximize_acquisition(sb::SequentialBatchAM, acq::AcquisitionFunction, problem::BossProblem, options::BossOptions)
    problem_ = deepcopy(problem)
    Xs = reduce(hcat, (speculative_evaluation!(problem_, sb.am, acq; options) for _ in 1:sb.batch_size))
    return Xs
end

function speculative_evaluation!(problem::BossProblem, am::AcquisitionMaximizer, acq::AcquisitionFunction; options::BossOptions)
    posterior = model_posterior(problem.model, problem.data)
    x = maximize_acquisition(am, acq, problem, options)
    y = posterior(x)[1]
    add_data!(problem.data, x, y)
    return x
end
