
"""
Return a vector of sample counts for each task,
so that `samples` samples are sampled in total among all tasks.
"""
function get_sample_counts(samples::Int, tasks::Int)
    base = floor(samples / tasks) |> Int
    diff = samples - (tasks * base)
    counts = Vector{Int}(undef, tasks)
    counts[1:diff] .= base + 1
    counts[diff+1:end] .= base
    return counts
end
