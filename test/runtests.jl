using Test
using BOSS
using Distributions
using LinearAlgebra

@testset "BOSS.jl" begin
    include("acquisition.jl")
    include("parametric.jl")
    include("nonparametric.jl")
    include("semiparametric.jl")
    include("term_cond.jl")
end
