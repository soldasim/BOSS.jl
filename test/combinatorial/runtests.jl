using OptimizationPRIMA

include("input_values.jl")
include("dummy_problem.jl")
include("file_utils.jl")

@testset "Combinatorial Tests" begin
    @info "Running Combinatorial Tests ..."
    
    inputs = load_input_coverage()
    for i in eachindex(inputs)
        val = get_input_vals(inputs[i])
        
        @testset "Test $i" begin
            script = create_problem(val)
            
            if val("VALID")
                if ismissing(val("f"))
                    @test script() isa AbstractVector{<:Real}
                else
                    @test script() isa BOSS.BossProblem
                end
            else
                @test_throws Exception script()
            end
        end
    end
end
