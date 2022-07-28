using JLD2

# DATA STRUCTS

struct RunParams{N}
    iters::Int
    init_X::Matrix{N}
    init_Y::Matrix{N}
    sample_count::Int
    util_opt_multistart::Int
end

struct RunResult{N}
    time::N
    X::Matrix{N}
    Y::Vector{N}
    bsf::Vector{N}
    errs::Union{Nothing, Vector{N}}
end

# LOAD/SAVE FUNCTIONS

function save_data(data, file_name)
    @save file_name data
end
function save_data(data, dir, file_name)
    return save_data(data, dir * file_name)
end

function load_data(file_name)
    data = nothing
    @load file_name data
    return data
end
function load_data(dir, file_name)
    return load_data(dir * file_name)
end
