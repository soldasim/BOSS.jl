
"""
Maximizes the acquisition function by simply checking a fine grid of points from the domain.

Extremely simple optimizer which can be used for simple problems or for debugging.
Not suitable for problems with high dimensional domain.

Can be used with constraints on `x`.
"""
struct GridAM <: AcquisitionMaximizer
    points::Vector{Vector{Float64}}
    steps::Vector{Float64}
    parallel::Bool
end
function GridAM(;
    problem::BOSS.OptimizationProblem,
    steps::Vector{Float64},
    parallel=true,
)
    domain = problem.domain
    ranges = [domain.bounds[1][i]:steps[i]:domain.bounds[2][i] for i in 1:x_dim(problem)]
    points = [[x...] for x in Iterators.product(ranges...) if in_domain(domain, [x...])]
    return GridAM(points, steps, parallel)
end

function maximize_acquisition(optimizer::GridAM, problem::BOSS.OptimizationProblem, acq::Function; info::Bool)
    if optimizer.parallel
        args = Vector{Vector{Float64}}(undef, Threads.nthreads())
        vals = fill(0., Threads.nthreads())

        Threads.@threads for p in optimizer.points
            v = acq(p)
            if (v > vals[Threads.threadid()])
                args[Threads.threadid()] = p
                vals[Threads.threadid()] = v
            end
        end
        
        return args[argmax(vals)]

    else
        return argmax(acq, optimizer.points)
    end
end
