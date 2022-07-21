using Distributions
using Soss

function m_poly(x, a, b, c, d)
    return a + b * x[1] + c * x[1]^2 + d * x[1]^3
end

function prob_model_poly()
    return @model X begin
            a ~ Distributions.Normal(1., 1.)
            b ~ Distributions.Normal(1., 1.)
            c ~ Distributions.Normal(1., 1.)
            d ~ Distributions.Normal(1., 1.)

            σ ~ Distributions.Exponential()
            Y ~ For(X) do x
                    Distributions.Normal(m_poly(x, a, b, c, d), σ)
                end
            return Y
        end
end

function model_params_poly()
    return [:a, :b, :c, :d]
end

function model_poly()
    return SSModel(m_poly, prob_model_poly(), model_params_poly())
end
