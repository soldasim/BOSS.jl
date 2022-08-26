using Distributions
using Turing

# The model 'y = a * exp(b * x) * cos(c * x + d) + e' is defined below.

const expcos_param_count_ = 5

function expcos_predict_(x, params)
    return [params[1] * exp(params[2] * x[1]) * safe_cos_(params[3] * x[1] + params[4]) + params[5]]
end
function safe_cos_(x)
    isinf(x) && return 0.
    return cos(x)
end

function expcos_priors_()
    return [Normal(1., 1.) for _ in 1:expcos_param_count_]
end

function model_expcos()
    return Boss.NonlinModel(
        expcos_predict_,
        expcos_priors_(),
        expcos_param_count_,
    )
end
