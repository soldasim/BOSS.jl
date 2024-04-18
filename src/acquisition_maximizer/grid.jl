
"""
    GridAM(problem, steps; kwargs...)

Maximizes the acquisition function by checking a fine grid of points from the domain.

Extremely simple optimizer which can be used for simple problems or for debugging.
Not suitable for problems with high dimensional domain.

Can be used with constraints on `x`.

# Arguments
- `problem::BOSS.BossProblem`: Provide your defined optimization problem.
- `steps::Vector{Float64}`: Defines the size of the grid gaps in each `x` dimension.

# Keywords
- `parallel::Bool`: If `parallel=true` then the optimization is parallelized. Defaults to `true`.
"""
struct GridAM <: AcquisitionMaximizer
    points::Vector{Vector{Float64}}
    steps::Vector{Float64}
    parallel::Bool
    shuffle::Bool
end
function GridAM(;
    problem::BOSS.BossProblem,
    steps,
    parallel=true,
    shuffle=true,
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
    if opt.parallel
        vals = Vector{Float64}(undef, length(points_))
        Threads.@threads for i in eachindex(points_)
            p = points_[i]
            v = acq(p)
            vals[i] = v
        end
        return points_[argmax(vals)]
    else
        return argmax(acq, points_)
    end
end
