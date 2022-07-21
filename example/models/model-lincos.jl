using Distributions
using Soss

function m_lincos(x, a, b, c, d)
    return a * x[1] * safe_cos(b * x[1] + c) + d
end
function safe_cos(x)
    isinf(x) && return 0
    return cos(x)
end

function prob_model_lincos()
    return @model X begin
            a ~ Distributions.Normal(1., 1.)
            b ~ Distributions.Normal(1., 1.)
            c ~ Distributions.Normal(1., 1.)
            d ~ Distributions.Normal(1., 1.)

            σ ~ Distributions.Exponential()
            Y ~ For(X) do x
                    Distributions.Normal(m_lincos(x, a, b, c, d), σ)
                end
            return Y
        end
end

function model_params_lincos()
    return [:a, :b, :c, :d]
end

function model_lincos()
    return SSModel(m_lincos, prob_model_lincos(), model_params_lincos())
end
