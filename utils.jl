using Distributions

# Sample from uniform distribution.
function uniform_sample(a, b, sample_size)
    distr = Product(Uniform.(a, b))
    X = rand(distr, sample_size)
    return vec.(collect.(eachslice(X; dims=length(size(X)))))
end

# Sample from log-uniform distribution.
function log_sample(a, b, sample_size)
    X = [exp.(x) for x in uniform_sample(log.(a), log.(b), sample_size)]
    return X
end

# Calculate RMS error with given test data.
function rms_error(X, Y, model)
    N = size(X)[1]
    preds = model.(eachrow(X))
    return sqrt((1 / N) * sum((preds - Y).^2))
end

# Calculate RMS error using uniformly sampled test data.
function rms_error(obj_func, model, a, b, sample_count)
    X = uniform_sample(a, b, sample_count)
    Y = reduce(vcat, reduce(vcat, obj_func.(X)))
    return rms_error(X, Y, model)
end

# Optimize function f by evaluating evenly distributed points over its domain.
function arg_opt(f, a, b; samples=2000)
    x_set = collect(LinRange(a, b, samples))
    y_set = f.(x_set)
    opt_i = argmax(y_set)

    return x_set[opt_i], y_set[opt_i], x_set, y_set
end

# Return points distributed evenly over a given logarithmic range.
function log_range(a, b, len)
     a = log10.(a)
     b = log10.(b)
     range = collect(LinRange(a, b, len))
     range = [10 .^ range[i] for i in 1:len]
     return range
end

# Return a copy of the given array without 'nothing' values.
function skipnothing(array)
    return [e for e in array if !isnothing(e)]
end
