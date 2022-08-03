using Distributions
using Soss

function m_exp_(x, params)
    return params[1] * exp(params[2] * x[1]) + params[3]
end

function prob_model_exp_()
    return @model X, noise begin
        params ~ For(zeros(3)) do _
            Distributions.Normal(1., 1.)
        end

        Y ~ For(collect(eachrow(X))) do x
            Distributions.Normal(m_exp_(x, params), noise[1])
        end
        return Y
    end
end

function model_exp()
    return SSModel(m_exp_, prob_model_exp_())
end
