
@testset "Unit Tests" begin
    include("acquisition_function/expected_improvement.jl")
    include("models/parametric.jl")
    include("models/nonparametric.jl")
    include("models/semiparametric.jl")
    include("term_cond.jl")
end
