module FileUtils

export load_var_names
export load_input_coverage

const COMBINATIONS_FILE = "./combinatorial/combinations.csv"

"""
Load a list of input variable names.
"""
function load_var_names()
    line = get_first_line(COMBINATIONS_FILE)
    @assert !isnothing(line)
    names = split(line, ',')
    return names
end

"""
Load combinations of input value names from the csv file.

Return a vector of dictionaries 'var_name -> value'.
"""
function load_input_coverage()
    f = open(COMBINATIONS_FILE, "r")
    names = nothing
    inputs = nothing

    idx = 0
    while !eof(f)
        line = format(readline(f))
        isnothing(line) && continue
        vals = split(line, ',')
        
        if idx == 0
            names = copy(vals)
            inputs = Dict{String, String}[]

        else
            d = Dict{String, String}()
            for i in eachindex(names)
                d[names[i]] = vals[i]
            end
            push!(inputs, d)
        end
        
        idx += 1
    end

    close(f)
    return inputs
end

"""
Get first non-skip line. Return `nothing` if no such line is present.
"""
function get_first_line(file)
    f = open(file, "r")
    line = nothing
    while !eof(f)
        line = format(readline(f))
        isnothing(line) || break
    end
    close(f)
    return line
end

"""
Remove whitespace and return `nothing` iff the line should be skipped.

Empty lines and comments are skipped.
"""
function format(line)
    line = filter(c -> !isspace(c), line)
    (length(line) == 0) && return nothing
    (first(line) == '#') && return nothing
    return line
end

end # module FileUtils
