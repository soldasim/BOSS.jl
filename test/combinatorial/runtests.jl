using UnitTestDesign
using OptimizationPRIMA
using Turing
using Distributions
using KernelFunctions
using LinearAlgebra
using Bijectors

include("builders.jl")
include("parameters.jl")

@testset "Combinatorial Tests" begin
    @info "Running Combinatorial Tests ..."
    test_cases = generate_test_cases()
    
    for i in eachindex(test_cases)
        case, run_case = test_cases[i]
        f = case[2]

        @testset "Test $i" begin
            if ismissing(f)
                @test run_case() isa AbstractVector{<:Real}
            else
                @test run_case() isa BossProblem
            end
        end
    end
end
