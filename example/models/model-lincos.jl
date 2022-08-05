using Distributions

const lincos_param_count_ = 4

function lincos_predict_(x, params)
    return [params[1] * x[1] * safe_cos_(params[2] * x[1] + params[3]) + params[4]]
end
function safe_cos_(x::Real)
    isinf(x) && return 0.
    return cos(x)
end

@model function lincos_prob_model_(X, Y, noise)
    params ~ MvNormal(ones(lincos_param_count_), ones(lincos_param_count_))

    for i in 1:size(X)[1]
        Y[i,:] ~ MvNormal(lincos_predict_(X[i,:], params), noise)
    end
end

function model_lincos()
    return SSModel(lincos_predict_, lincos_prob_model_, lincos_param_count_)
end
