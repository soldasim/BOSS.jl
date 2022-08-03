using Soss

struct SSModel
    predict # y = predict(x, params...)
    prob_model # Soss.@model
end

# TODO refactor: params and noise HAVE TO be called 'params', 'noise' for now



# - - - - - - - - EXAMPLE MODEL DEFINITION - - - - - - - - - - - - - - - -

## The model 'f(x) = [a*cos(b*x[1])+c, d*sin(e*x[1])+f]' is defined below.


## Start by defining the predictive function.

# const param_count = 6
# const y_dim = 2

# function m_predict_(x, params)
#     y1 = params[1] * cos(params[2] * x[1]) + params[3]
#     y2 = params[4] * cos(params[5] * x[1]) + params[6]
#     return [y1, y2]
# end


## Now define the probabilstic model used for HMC sampling.
## The labeling 'params' and 'noise' have to be kept! (for now)

# function m_prob_()
#     return @model X begin
#         params ~ For(zeros(param_count)) do _
#             Distributions.Normal(1., 1.)
#         end
#
#         noise ~ For(zeros(y_dim)) do _
#             Distributions.Exponential()
#         end
#
#         Y ~ For(collect(eachrow(X))) do x
#             For(m_predict_(x, params), noise) do yi, ni
#                 Distributions.Normal(yi, ni)
#             end
#         end
#         return Y
#     end
# end


## Retrieve the model using the following function.

# model() = SSModel(
#         m_predict_,
#         m_prob_(),
#     )
