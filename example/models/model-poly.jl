using Distributions
using Soss

function m_poly_(x, params)
    return params[1] + params[2] * x[1] + params[3] * x[1]^2 + params[4] * x[1]^3
end

function prob_model_poly_()
    return @model X begin
        params ~ For(zeros(4)) do _
            Distributions.Normal(1., 1.)
        end

        noise ~ For(zeros(1)) do _
            Distributions.Exponential()
        end

        Y ~ For(collect(eachrow(X))) do x
            Distributions.Normal(m_poly_(x, params), noise[1])
        end
        return Y
    end
end

function model_poly()
    return SSModel(m_poly_, prob_model_poly_())
end
