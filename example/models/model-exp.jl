using Distributions
using Soss

function m_exp(x, a, b, c)
    return a * exp(b * x[1]) + c
end

function prob_model_exp()
    return @model X begin
            a ~ Distributions.Normal(1., 1.)
            b ~ Distributions.Normal(1., 1.)
            c ~ Distributions.Normal(1., 1.)

            σ ~ Distributions.Exponential()
            Y ~ For(X) do x
                    Distributions.Normal(m_exp(x, a, b, c), σ)
                end
            return Y
        end
end

function model_params_exp()
    return [:a, :b, :c]
end

function model_exp()
    return SSModel(m_exp, prob_model_exp(), model_params_exp())
end
