using Distributions
using Soss

function m_sq_(x, params)
    return params[1] + params[2] * x[1] + params[3] * x[1]^2
end

function prob_model_sq_()
    return @model X, noise begin
        params ~ For(zeros(3)) do _
            Distributions.Normal(1., 1.)
        end

        Y ~ For(collect(eachrow(X))) do x
            Distributions.Normal(m_sq_(x, params), noise[1])
        end
        return Y
    end
end

function model_sq()
    return SSModel(m_sq_, prob_model_sq_())
end
