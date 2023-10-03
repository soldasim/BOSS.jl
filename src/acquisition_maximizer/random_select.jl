
"""
Selects a random interior point instead of maximizing the acquisition function.
Can be used for method comparison.

Can handle constraints on `x`, but does so by generating random points in the box domain
until a point satisfying the constraints is found. Therefore it can take a long time
or even get stuck if the constraints are very tight.
"""
struct RandomSelectAM <: AcquisitionMaximizer end

function maximize_acquisition(acq::Function, optimizer::RandomSelectAM, problem::BOSS.OptimizationProblem, options::BossOptions)
    domain = problem.domain
    x = random_start(domain.bounds)
    if !isnothing(domain.cons)
        while any(domain.cons(x) .< 0.)
            x = random_start(domain.bounds)
        end
    end
    return x
end
