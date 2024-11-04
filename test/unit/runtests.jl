#using ParamTests  # TODO
include("_parametrized_tests.jl")  # TODO

function include_unit_tests(dir)
    for file in readdir(dir)
        if endswith(file, ".jl")
            @info "  Testing $dir/$file"
            @testset "$file" begin
                include(dir*'/'*file)
            end
        else
            @testset "$file" begin
                include_unit_tests(dir*'/'*file)
            end
        end
    end
end

@testset "Unit Tests" begin
    @info "Running Unit Tests ..."
    include_unit_tests(Base.source_dir() * '/' * "test")
end
