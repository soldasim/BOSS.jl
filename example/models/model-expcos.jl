using Distributions

const expcos_param_count_ = 5

function expcos_predict_(x, params)
    return [params[1] * exp(params[2] * x[1]) * safe_cos_(params[3] * x[1] + params[4]) + params[5]]
end
function safe_cos_(x)
    isinf(x) && return 0.
    return cos(x)
end

@model function expcos_prob_model_(X, Y, noise)
    params ~ MvNormal(ones(expcos_param_count_), ones(expcos_param_count_))

    for i in 1:size(X)[1]
        Y[i,:] ~ MvNormal(expcos_predict_(X[i,:], params), noise)
    end
end

function model_expcos()
    return SSModel(expcos_predict_, expcos_prob_model_, expcos_param_count_)
end
