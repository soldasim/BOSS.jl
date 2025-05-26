
"""
    RandomAM()

Selects a random interior point instead of maximizing the acquisition function.
Can be used for method comparison.

Can handle constraints on `x`, but does so by generating random points in the box domain
until a point satisfying the constraints is found. Therefore it can take a long time
or even get stuck if the constraints are very tight.
"""
struct RandomAM <: AcquisitionMaximizer end

function maximize_acquisition(::RandomAM, problem::BossProblem, options::BossOptions)
    domain = problem.domain
    x = random_x(domain)
    if !isnothing(domain.cons)
        while !in_cons(x, domain.cons)
            x = random_x(domain)
        end
    end
    return x, nothing
end

function random_x(domain)
    x = random_point(domain.bounds)
    x = cond_func(round).(x, domain.discrete)  # assure discrete dims
    return x
end
