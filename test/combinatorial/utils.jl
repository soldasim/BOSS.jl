
function load_input_coverage()
    f = open("./combinatorial/combinations.csv", "r")
    names = nothing
    inputs = nothing

    idx = 0
    while !eof(f)
        line = readline(f)
        line = filter(c -> !isspace(c), line)
        (length(line) == 0) && continue
        (first(line) == '#') && continue
        vals = split(line, ',')
        
        if idx == 0
            names = copy(vals)
            inputs = Dict{String, String}[]
            assert_input_names(names)
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

function assert_input_names(names)
    for var in names
        @assert var in INPUT_NAMES
    end
end
