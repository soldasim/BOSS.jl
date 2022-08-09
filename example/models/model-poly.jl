using Distributions
using Turing

# The model 'y = a + b*x + c*x^2 + d*x^3' is defined below.

function poly_lift_(x)
    ϕ1 = [
        1.,
        x[1],
        x[1]^2,
        x[1]^3,
    ]
    return [ϕ1]
end

function poly_priors_()
    p1 = (zeros(4), Diagonal(10. * ones(4)))
    return [p1]
end

function model_poly()
    return Boss.LinModel(poly_lift_, poly_priors_())
end
