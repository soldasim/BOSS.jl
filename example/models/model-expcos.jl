using Distributions
using Soss

function m_expcos_(x, params)
    return params[1] * exp(params[2] * x[1]) * safe_cos_(params[3] * x[1] + params[4]) + params[5]
end
function safe_cos_(x)
    isinf(x) && return 0.
    return cos(x)
end

function prob_model_expcos_()
    return @model X, noise begin
        params ~ For(zeros(5)) do _
            Distributions.Normal(1., 1.)
        end

        Y ~ For(collect(eachrow(X))) do x
            Distributions.Normal(m_expcos_(x, params), noise[1])
        end
        return Y
    end
end

function model_expcos()
    return SSModel(m_expcos_, prob_model_expcos_())
end
