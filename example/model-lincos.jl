using Distributions
using Soss

function m(x, a, b, c, d)
    return a * x[1] * safe_cos(b * x[1] + c) + d
end
function safe_cos(x)
    isinf(x) && return 0
    return cos(x)
end

function get_prob_model()
    return @model X begin
            a ~ Distributions.Normal(1., 1.)
            b ~ Distributions.Normal(1., 1.)
            c ~ Distributions.Normal(1., 1.)
            d ~ Distributions.Normal(1., 1.)

            σ ~ Distributions.Exponential()
            Y ~ For(X) do x
                    Distributions.Normal(m(x, a, b, c, d), σ)
                end
            return Y
        end
end

function get_model_params()
    return [:a, :b, :c, :d]
end

function get_model()
    return ModelSS(m, get_prob_model(), get_model_params())
end
