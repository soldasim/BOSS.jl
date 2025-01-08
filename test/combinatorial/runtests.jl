using OptimizationPRIMA

include("file_utils.jl")
include("input_values.jl")
include("dummy_problem.jl")

using .FileUtils
using .InputValues

@testset "Combinatorial Tests" begin
    @info "Running Combinatorial Tests ..."
    
    inputs = load_input_coverage()
    for i in eachindex(inputs)
        val = get_input_vals(inputs[i])
        
        @testset "Test $i" begin
            script = create_problem(val)
            
            if ismissing(val("f"))
                @test script() isa AbstractVector{<:Real}
            else
                @test script() isa BossProblem
            end
        end
    end
end
