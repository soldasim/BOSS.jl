using Test
using BOSS

@testset "BOSS.jl" begin
    include("unit/acquisition_function/expected_improvement.jl")
    include("unit/models/parametric.jl")
    include("unit/models/nonparametric.jl")
    include("unit/models/semiparametric.jl")
    include("unit/term_cond.jl")
end
