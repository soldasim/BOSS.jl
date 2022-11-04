using JLD2

# DATA STRUCTS

struct RunResult
    time
    X
    Y
    Z
    bsf
    parameters
    errs
end

# ## OLD (needed for loading old run-data)
# struct RunResult
#     time
#     X
#     Y
#     Z
#     bsf
#     errs
# end

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
