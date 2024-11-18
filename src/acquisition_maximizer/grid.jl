
"""
    GridAM(kwargs...)

Maximizes the acquisition function by checking a fine grid of points from the domain.

Extremely simple optimizer which can be used for simple problems or for debugging.
Not suitable for problems with high dimensional domain.

Can be used with constraints on `x`.

# Keywords
- `problem::BossProblem`: Provide your defined optimization problem.
- `steps::Vector{Float64}`: Defines the size of the grid gaps in each `x` dimension.
- `parallel::Bool`: If `parallel=true`, the optimization is parallelized. Defaults to `true`.
- `shuffle::Bool`: If `shuffle=true`, the grid points are shuffled before each optimization. Defaults to `true`.
"""
struct GridAM <: AcquisitionMaximizer
    points::Vector{Vector{Float64}}
    steps::Vector{Float64}
    parallel::Bool
    shuffle::Bool
end
function GridAM(;
    problem::BossProblem,
    steps,
    parallel = true,
    shuffle = true,
)
    domain = problem.domain
    ranges = dim_range.(domain.bounds..., steps, domain.discrete)
    points = [[x...] for x in Iterators.product(ranges...) if in_domain([x...], domain)]
    return GridAM(points, steps, parallel, shuffle)
end

function dim_range(lb, ub, step, discrete)
    if discrete
        isinteger(step) || @warn "Non-integer `step` provided for a discrete dimension! Rounding `step` from $(step) up to $(ceil(step))."
        return ceil(lb):ceil(step):floor(ub)
    else
        return lb:step:ub
    end
end

function maximize_acquisition(opt::GridAM, acquisition::AcquisitionFunction, problem::BossProblem, options::BossOptions)
    acq = acquisition(problem, options)
    points_ = opt.shuffle ? (opt.points |> deepcopy |> shuffle) : opt.points
    x, val = optimize(Val(opt.parallel), opt, acq, points_)
    return x, val
end

function optimize(parallel::Val{false}, opt::GridAM, acq, points::Vector{Vector{Float64}})
    best = argmax(acq, points)
    return best, acq(best)
end
function optimize(parallel::Val{true}, opt::GridAM, acq, points::Vector{Vector{Float64}})
    vals = Vector{Float64}(undef, length(points))
    Threads.@threads for i in eachindex(points)
        p = points[i]
        v = acq(p)
        vals[i] = v
    end
    best = argmax(vals)
    return points[best], vals[best]
end
