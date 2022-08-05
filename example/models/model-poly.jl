using Distributions

const poly_param_count_ = 4

function poly_predict_(x, params)
    return [params[1] + params[2] * x[1] + params[3] * x[1]^2 + params[4] * x[1]^3]
end

@model function poly_prob_model_(X, Y, noise)
    params ~ MvNormal(ones(poly_param_count_), ones(poly_param_count_))

    for i in 1:size(X)[1]
        Y[i,:] ~ MvNormal(poly_predict_(X[i,:], params), noise)
    end
end

function model_poly()
    return SSModel(poly_predict_, poly_prob_model_, poly_param_count_)
end
