using Soss

struct ModelSS
    predict::Function # y = predict(x, params...)
    prob_model::Model # Soss.@model
    params::AbstractArray{Symbol} # [:a, :b, :c, ...]
end
