using Distributions
using Turing

# The model 'y = a + b*x + c*x^2 + d*x^3' is defined below.

const poly_param_count_ = 4

function poly_lift_(x)
    ϕ1 = [
        1.,
        x[1],
        x[1]^2,
        x[1]^3,
    ]
    return [ϕ1]
end

function poly_priors_()
    return [Normal(1., 1.) for _ in 1:poly_param_count_]
end

function model_poly()
    return Boss.LinModel(
        poly_lift_,
        poly_priors_(),
        poly_param_count_,
    )
end
