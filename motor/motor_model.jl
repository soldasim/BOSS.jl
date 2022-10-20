using Distributions

# LINMODEL

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

const motor_param_count_ = 30

motor_priors_() = [Normal(0., 1.) for _ in 1:motor_param_count_]

motor_model() = Boss.LinModel(
    motor_lift_,
    motor_priors_(),
    motor_param_count_
)

# NONLINMODEL

function motor_predict_(x, params)
    ϕ = [
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
    ]
    θ1, θ2, θ3 = params[1:10], params[11:20], params[21:30]
    return sum.([θ1 .* ϕ, θ2 .* ϕ, θ3 .* ϕ])
end

motor_model() = Boss.NonlinModel(
    motor_predict_,
    motor_priors_(),
    motor_param_count_
)
