# TODO docs, comments & example

# - - - - - - - - FITNESS - - - - - - - - - - - - - - - -

abstract type Fitness end

"""
Used to define a linear fitness function for the BOSS algorithm.

# Example
A fitness function 'f(y) = y[1] + a * y[2] + b * y[3]' can be defined as:
```julia-repl
julia> LinFitness([1., a, b])
```
"""
struct LinFitness <: Fitness
    coefs
end
function (f::LinFitness)(y)
    return f.coefs' * y
end

"""
Used to define a fitness function for the BOSS algorithm.
If possible, the 'LinFitness' option should be used instead for a better performance.

# Example
```julia-repl
julia> NonlinFitness(y -> cos(y[1]) + sin(y[2]))
```
"""
struct NonlinFitness <: Fitness
    fitness
end
function (f::NonlinFitness)(y)
    return f.fitness(y)
end

# - - - - - - - - MODEL - - - - - - - - - - - - - - - -

abstract type ParamModel end

"""
Used to define a linear parametric model for the BOSS algorithm.
The model has to be linear in its parameters and have Gaussian parameter priors.
This model definition provides better performance than the 'NonlinModel' option.
"""
struct LinModel <: ParamModel
    lift
    param_priors
end

"""
Used to define a parametric model for the BOSS algorithm.
If possible, the 'LinModel' option should be used instead for a better performance.
"""
struct NonlinModel <: ParamModel
    predict
    prob_model
    param_count
end
