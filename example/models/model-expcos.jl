using Distributions
using Soss

function m_expcos(x, a, b, c, d, e)
    return a * exp(b * x[1]) * safe_cos(c * x[1] + d) + e
end
function safe_cos(x)
    isinf(x) && return 0
    return cos(x)
end

function prob_model_expcos()
    return @model X begin
            a ~ Distributions.Normal(1., 1.)
            b ~ Distributions.Normal(1., 1.)
            c ~ Distributions.Normal(1., 1.)
            d ~ Distributions.Normal(1., 1.)
            e ~ Distributions.Normal(1., 1.)

            σ ~ Distributions.Exponential()
            Y ~ For(X) do x
                    Distributions.Normal(m_expcos(x, a, b, c, d, e), σ)
                end
            return Y
        end
end

function model_params_expcos()
    return [:a, :b, :c, :d, :e]
end

function model_expcos()
    return SSModel(m_expcos, prob_model_expcos(), model_params_expcos())
end
