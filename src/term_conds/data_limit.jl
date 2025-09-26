
"""
    DataLimit(data_max::Int)

Terminates the BOSS algorithm after a predefined number of data is collected.
(Once `size(problem.data.X, 2) == data_max` the algorithm stops.)

See also: [`bo!`](@ref)
"""
struct DataLimit <: TermCond
    data_max::Int
end

function (cond::DataLimit)(problem::BossProblem)
    size(problem.data.X, 2) < cond.data_max
end
