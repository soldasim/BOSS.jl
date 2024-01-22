```
Determines whether parallelization of BOSS is allowed during tests.

Set `PARALLEL_TESTS = true` to allow for BOSS parallelization to be tested.
```
const PARALLEL_TESTS = true

include("input_values.jl")
include("dummy_problem.jl")
include("utils.jl")

@testset "Combinatorial Tests" begin
    inputs = load_input_coverage()
    for i in eachindex(inputs)
        comb = inputs[i]
        
        @testset "Test $i" begin
            in(var) = INPUT_DICT[var][comb[var]]
            script = create_problem(in)
            
            if in("VALID")
                if ismissing(in("f"))
                    @test script() isa AbstractVector{<:Real}
                else
                    @test script() isa BOSS.OptimizationProblem
                end
            else
                @test_throws Exception script()
            end
        end
    end
end
