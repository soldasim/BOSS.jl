using Distributions

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

motor_model() = Boss.LinModel(motor_lift_, motor_priors_(), motor_param_count_)
