using Distributions
include("../model.jl")

motor_lift_(x) = [[
    x[1]^2,
    x[2]^2,
    x[3]^2,
    x[1]*x[2],
    x[2]*x[3],
    x[1]*x[3],
    x[1],
    x[2],
    x[3],
    1.,
] for _ in 1:3]

motor_priors_() = [(zeros(10), Diagonal(ones(10))) for _ in 1:3]

motor_model() = LinModel(motor_lift_, motor_priors_())
