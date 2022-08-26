using Distributions
using Turing

# The model 'y = a * x * cos(b * x + c) + d' is defined below.

const lincos_param_count_ = 4

function lincos_predict_(x, params)
    return [params[1] * x[1] * safe_cos_(params[2] * x[1] + params[3]) + params[4]]
end
function safe_cos_(x::Real)
    isinf(x) && return 0.
    return cos(x)
end

function lincos_priors_()
    return [Normal(1., 1.) for _ in 1:lincos_param_count_]
end

function model_lincos()
    return Boss.NonlinModel(
        lincos_predict_,
        lincos_priors_(),
        lincos_param_count_,
    )
end
