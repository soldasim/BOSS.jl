using Distributions
using Soss

function m_sq(x, a, b, c)
    return a + b * x[1] + c * x[1]^2
end

function prob_model_sq()
    return @model X begin
            a ~ Distributions.Normal(1., 1.)
            b ~ Distributions.Normal(1., 1.)
            c ~ Distributions.Normal(1., 1.)

            σ ~ Distributions.Exponential()
            Y ~ For(X) do x
                    Distributions.Normal(m_sq(x, a, b, c), σ)
                end
            return Y
        end
end

function model_params_sq()
    return [:a, :b, :c]
end

function model_sq()
    return SSModel(m_sq, prob_model_sq(), model_params_sq())
end
