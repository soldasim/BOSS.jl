#using ParamTests  # TODO

include("utils.jl")
include("_parametrized_tests.jl")  # TODO

@testset "Unit Tests" begin
    test_files = list_files_rec(Base.source_dir() * "\\test")
    include.(test_files)
end
