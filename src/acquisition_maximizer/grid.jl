
"""
    GridAM(problem, steps; kwargs...)

Maximizes the acquisition function by checking a fine grid of points from the domain.

Extremely simple optimizer which can be used for simple problems or for debugging.
Not suitable for problems with high dimensional domain.

Can be used with constraints on `x`.

# Arguments
- `problem::BOSS.OptimizationProblem`: Provide your defined optimization problem.
- `steps::Vector{Float64}`: Defines the size of the grid gaps in each `x` dimension.

# Keywords
- `parallel::Bool`: If `parallel=true` then the optimization is parallelized. Defaults to `true`.
"""
struct GridAM <: AcquisitionMaximizer
    points::Vector{Vector{Float64}}
    steps::Vector{Float64}
    parallel::Bool
end
function GridAM(;
    problem::BOSS.OptimizationProblem,
    steps,
    parallel=true,
)
    domain = problem.domain
    ranges = [domain.bounds[1][i]:steps[i]:domain.bounds[2][i] for i in 1:x_dim(problem)]
    points = [[x...] for x in Iterators.product(ranges...) if in_domain([x...], domain)]
    return GridAM(points, steps, parallel)
end

function maximize_acquisition(acq::Function, opt::GridAM, problem::BOSS.OptimizationProblem, options::BossOptions)
    if opt.parallel
        vals = Vector{Float64}(undef, length(opt.points))
        Threads.@threads for i in eachindex(opt.points)
            p = opt.points[i]
            v = acq(p)
            vals[i] = v
        end
        return opt.points[argmax(vals)]
    else
        return argmax(acq, opt.points)
    end
end
