using Test
using BOSS

@testset "BOSS TESTS" verbose=true begin
    include("unit/runtests.jl")
    include("combinatorial/runtests.jl")
end
