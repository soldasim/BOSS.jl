
"""
    bounds = ([0, 0], [1, 1])

const AbstractBounds = Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}

Defines box constraints.
"""
const AbstractBounds = Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}


# - - - - - - - - Acquisition Functions - - - - - - - -

"""
Specifies the acquisition function describing the "quality" of a potential next evaluation point.
Inherit this type to define a custom acquisition function.

Example: `struct CustomAcq <: AcquisitionFunction ... end`

Structures derived from this type have to implement the following method:
`(acquisition::CustomAcq)(problem::BossProblem, options::BossOptions)`

This method should return a function `acq(x::AbstractVector{<:Real}) = val::Real`,
which is maximized to select the next evaluation function of blackbox function in each iteration.

See also: [`ExpectedImprovement`](@ref)
"""
abstract type AcquisitionFunction end

# Specific implementations of `AcquisitionFunction` are in '\src\acquisition'.


# - - - - - - - - Surrogate Model - - - - - - - -

"""
An abstract type for a surrogate model approximating the objective function.

Example usage: `struct CustomModel <: SurrogateModel ... end`

All models should implement the following methods:
- `make_discrete(model::CustomModel, discrete::AbstractVector{<:Bool}) -> discrete_model::CustomModel`
- `model_posterior(model::CustomModel, data::ExperimentDataMAP; split::Bool) -> (x -> mean, std) <or> [(x -> mean_i, std_i) for i in 1:y_dim]`
- `model_posterior(model::CustomModel, data::ExperimentDataBI; split::Bool) -> [(x -> mean, std) for each sample] <or> [[(x -> mean_i, std_i) for i in 1:y_dim] for each sample]`
- `model_loglike(model::CustomModel, noise_std_priors::AbstractVector{<:UnivariateDistribution}, data::ExperimentData) -> (θ, length_scales, noise_std -> loglike)`
- `sample_params(model::CustomModel, noise_std_priors::AbstractVector{<:UnivariateDistribution}) -> (θ::AbstractVector{<:Real}, λ::AbstractMatrix{<:Real}, noise_std::AbstractVector{<:Real})`
- `param_priors(model::CustomModel) -> (θ_priors::AbstractVector{<:UnivariateDistribution}, λ_priors::AbstractVector{<:MultivariateDistribution})`

See also:
[`LinModel`](@ref), [`NonlinModel`](@ref),
[`GaussianProcess`](@ref),
[`Semiparametric`](@ref)
"""
abstract type SurrogateModel end

# Specific implementations of `SurrogateModel` are in '\src\models'.


# - - - - - - - - Acquisition Maximization - - - - - - - -

"""
Specifies the library/algorithm used for acquisition function optimization.
Inherit this type to define a custom acquisition maximizer.

Example: `struct CustomAlg <: AcquisitionMaximizer ... end`

Structures derived from this type have to implement the following method:
`maximize_acquisition(acq_maximizer::CustomAlg, acq::AcquisitionFunction, problem::BossProblem, options::BossOptions)`
This method should return the point of the input domain which maximizes the given acquisition function `acq` (as a vector)
or a batch of points (as a column-wise matrix).

See also: [`OptimizationAM`](@ref)
"""
abstract type AcquisitionMaximizer end

# Specific implementations of `AcquisitionMaximizer` are in '\src\acquisition_maximizer'.


# - - - - - - - - Model-Fitting - - - - - - - -

"""
An abstract type used to differentiate between
MAP (Maximum A Posteriori) optimizers and BI (Bayesian Inference) samplers.
"""
abstract type ModelFit end
struct MAP <: ModelFit end
struct BI <: ModelFit end

"""
Specifies the library/algorithm used for model parameter estimation.
Inherit this type to define a custom model-fitting algorithms.

Example: `struct CustomFitter <: ModelFitter{MAP} ... end` or `struct CustomFitter <: ModelFitter{BI} ... end`

Structures derived from this type have to implement the following method:
`estimate_parameters(model_fitter::CustomFitter, problem::BossProblem; info::Bool)`.

This method should return a named tuple `(θ = ..., length_scales = ..., noise_std = ...)`
with either MAP model parameters (if `CustomAlg <: ModelFitter{MAP}`)
or model parameter samples (if `CustomAlg <: ModelFitter{BI}`).

See also: [`OptimizationMAP`](@ref), [`TuringBI`](@ref)
"""
abstract type ModelFitter{T<:ModelFit} end

# Specific implementations of `ModelFitter` are in '\src\model_fitter'.


# - - - - - - - - Termination Conditions - - - - - - - -

"""
Specifies the termination condition of the whole BOSS algorithm.
Inherit this type to define a custom termination condition.

Example: `struct CustomCond <: TermCond ... end`

Structures derived from this type have to implement the following method:
`(cond::CustomCond)(problem::BossProblem)`

This method should return true to keep the optimization running
and return false once the optimization is to be terminated.

See also: [`IterLimit`](@ref)
"""
abstract type TermCond end

# Specific implementations of `TermCond` are in '\src\term_cond.jl'.


# - - - - - - - - Data - - - - - - - -

"""
Stores all the data collected during the optimization
as well as the parameters and hyperparameters of the model.

See also: [`ExperimentDataPrior`](@ref), [`ExperimentDataPost`](@ref)
"""
abstract type ExperimentData end
ExperimentData(args...) = ExperimentDataPrior(args...)

Base.length(data::ExperimentData) = size(data.X)[2]
Base.isempty(data::ExperimentData) = isempty(data.X)

"""
Stores the initial data.

# Fields
- `X::AbstractMatrix{<:Real}`: Contains the objective function inputs as columns.
- `Y::AbstractMatrix{<:Real}`: Contains the objective function outputs as columns.

See also: [`ExperimentDataPost`](@ref)
"""
mutable struct ExperimentDataPrior{
    T<:AbstractMatrix{<:Real},
} <: ExperimentData
    X::T
    Y::T
end
ExperimentDataPrior(;
    X,
    Y,
) = ExperimentDataPrior(X, Y)

"""
Stores the fitted/samples model parameters in addition to the data matrices `X`,`Y`.

See also: [`ExperimentDataPrior`](@ref), [`ExperimentDataMAP`](@ref), [`ExperimentDataBI`](@ref)
"""
abstract type ExperimentDataPost{T<:ModelFit} <: ExperimentData end

"""
Stores the data matrices `X`,`Y` as well as the optimized model parameters and hyperparameters.

# Fields
- `X::AbstractMatrix{<:Real}`: Contains the objective function inputs as columns.
- `Y::AbstractMatrix{<:Real}`: Contains the objective function outputs as columns.
- `θ::Union{Nothing, <:AbstractVector{<:Real}}`: Contains the MAP parameters
        of the parametric model (or nothing if the model is nonparametric).
- `length_scales::Union{Nothing, <:AbstractMatrix{<:Real}}`: Contains the MAP length scales
        of the GP as a `x_dim`×`y_dim` matrix (or nothing if the model is parametric).
- `amplitudes::Union{Nothing, <:AbstractVector{<:Real}}`: Amplitudes of the GP.
- `noise_std::AbstractVector{<:Real}`: The MAP noise standard deviations of each `y` dimension.
- `consistent::Bool`: True iff the parameters (`θ`, `length_scales`, `amplitudes`, `noise_std`)
        have been fitted using the current dataset (`X`, `Y`).
        Is set to `consistent = false` after updating the dataset,
        and to `consistent = true` after re-fitting the parameters.

See also: [`ExperimentDataBI`](@ref)
"""
mutable struct ExperimentDataMAP{
    T<:AbstractMatrix{<:Real},
    P<:Union{Nothing, <:AbstractVector{<:Real}},
    L<:Union{Nothing, <:AbstractMatrix{<:Real}},
    A<:Union{Nothing, <:AbstractVector{<:Real}},
    N<:AbstractVector{<:Real},
} <: ExperimentDataPost{MAP}
    X::T
    Y::T
    θ::P
    length_scales::L
    amplitudes::A
    noise_std::N
    consistent::Bool
end

"""
Stores the data matrices `X`,`Y` as well as the sampled model parameters and hyperparameters.

# Fields
- `X::AbstractMatrix{<:Real}`: Contains the objective function inputs as columns.
- `Y::AbstractMatrix{<:Real}`: Contains the objective function outputs as columns.
- `θ::Union{Nothing, <:AbstractVector{<:AbstractVector{<:Real}}}`: Samples of parameters of the parametric model
        stored column-wise in a matrix (or nothing if the model is nonparametric).
- `length_scales::Union{Nothing, <:AbstractVector{<:AbstractMatrix{<:Real}}}`: Samples
    of the length scales of the GP as a vector of `x_dim`×`y_dim` matrices
    (or nothing if the model is parametric).
- `amplitudes::Union{Nothing, <:AbstractVector{<:AbstractVector{<:Real}}}`: Samples of the amplitudes of the GP.
- `noise_std::AbstractVector{<:AbstractVector{<:Real}}`: Samples of the noise standard deviations of each `y` dimension
        stored column-wise in a matrix.
- `consistent::Bool`: True iff the parameters (`θ`, `length_scales`, `amplitudes`, `noise_std`)
        have been fitted using the current dataset (`X`, `Y`).
        Is set to `consistent = false` after updating the dataset,
        and to `consistent = true` after re-fitting the parameters.

See also: [`ExperimentDataMAP`](@ref)
"""
mutable struct ExperimentDataBI{
    T<:AbstractMatrix{<:Real},
    P<:Union{Nothing, <:AbstractVector{<:AbstractVector{<:Real}}},
    L<:Union{Nothing, <:AbstractVector{<:AbstractMatrix{<:Real}}},
    A<:Union{Nothing, <:AbstractVector{<:AbstractVector{<:Real}}},
    N<:AbstractVector{<:AbstractVector{<:Real}},
} <: ExperimentDataPost{BI}
    X::T
    Y::T
    θ::P
    length_scales::L
    amplitudes::A
    noise_std::N
    consistent::Bool
end


# - - - - - - - - Optimization Problem - - - - - - - -

"""
An abstract type for a fitness function
measuring the quality of an output `y` of the objective function.

Fitness is used by the `AcquisitionFunction` to determine promising points for future evaluations.

See also: [`AcquisitionFunction`](@ref), [`NoFitness`](@ref), [`LinFitness`](@ref), [`NonlinFitness`](@ref)
"""
abstract type Fitness end

"""
    NoFitness()

Placeholder for problems with no defined fitness. Problems with `NoFitness`
can only be solved with `AcquisitionFunction` which does not use fitness.
"""
struct NoFitness <: Fitness end

"""
    LinFitness(coefs::AbstractVector{<:Real})

Used to define a linear fitness function 
measuring the quality of an output `y` of the objective function.

May provide better performance than the more general `NonlinFitness`
as some acquisition functions can be calculated analytically with linear fitness
functions whereas this may not be possible with a nonlinear fitness function.

See also: [`NonlinFitness`](@ref)

# Example
A fitness function `f(y) = y[1] + a * y[2] + b * y[3]` can be defined as:
```julia-repl
julia> LinFitness([1., a, b])
```
"""
struct LinFitness{
    C<:AbstractVector{<:Real},
} <: Fitness
    coefs::C
end
(f::LinFitness)(y) = f.coefs' * y

"""
    NonlinFitness(fitness::Function)

Used to define a general nonlinear fitness function
measuring the quality of an output `y` of the objective function.

If your fitness function is linear, use `LinFitness` instead for better performance.

See also: [`LinFitness`](@ref)

# Example
```julia-repl
julia> NonlinFitness(y -> cos(y[1]) + sin(y[2]))
```
"""
struct NonlinFitness <: Fitness
    fitness::Function
end
(f::NonlinFitness)(y) = f.fitness(y)

"""
    Domain(; kwargs...)

Describes the optimization domain.

# Keywords
- `bounds::AbstractBounds`: The basic box-constraints on `x`. This field is mandatory.
- `discrete::AbstractVector{<:Bool}`: Can be used to designate some dimensions
        of the domain as discrete.
- `cons::Union{Nothing, Function}`: Used to define arbitrary nonlinear constraints on `x`.
        Feasible points `x` must satisfy `all(cons(x) .> 0.)`. An appropriate acquisition
        maximizer which can handle nonlinear constraints must be used if `cons` is provided.
        (See [`AcquisitionMaximizer`](@ref).)
"""
struct Domain{
    B<:AbstractBounds,
    D<:AbstractVector{<:Bool},
    C<:Union{Nothing, Function},
}
    bounds::B
    discrete::D
    cons::C
end
function Domain(;
    bounds,
    discrete = fill(false, length(first(bounds))),
    cons = nothing,
)
    @assert length(bounds[1]) == length(bounds[2]) == length(discrete)
    return Domain(bounds, discrete, cons)
end

"""
    BossProblem(; kwargs...)

Defines the whole optimization problem for the BOSS algorithm.

# Problem Definition

    There is some (noisy) blackbox function `y = f(x) = f_true(x) + ϵ` where `ϵ ~ Normal`.

    We have some surrogate model `y = model(x) ≈ f_true(x)`
    describing our knowledge (or lack of it) about the blackbox function.

    We wish to find `x ∈ domain` such that `fitness(f(x))` is maximized
    while satisfying the constraints `f(x) <= y_max`.

# Keywords
- `fitness::Fitness`: The fitness function. See [`Fitness`](@ref).
- `f::Union{Function, Missing}`: The objective blackbox function.
- `domain::Domain`: The domain of `x`. See [`Domain`](@ref).
- `y_max`: The constraints on `y`. (See the definition above.)
- `model::SurrogateModel`: See [`SurrogateModel`](@ref).
- `noise_std_priors::AbstractVector{<:UnivariateDistribution}`: The prior distributions
        of the noise standard deviations of each `y` dimension.
- `data::ExperimentData`: The initial data of objective function evaluations.
        See [`ExperimentDataPrior`].

See also: [`bo!`](@ref)
"""
mutable struct BossProblem{
    F<:Any,
}
    fitness::Fitness
    f::F
    domain::Domain
    y_max::AbstractVector{<:Real}
    model::SurrogateModel
    noise_std_priors::AbstractVector{<:UnivariateDistribution}
    data::ExperimentData
end
BossProblem(;
    fitness=NoFitness(),
    f,
    domain,
    model,
    noise_std_priors,
    y_max = fill(Inf, length(noise_std_priors)),
    data,
) = BossProblem(fitness, f, domain, y_max, model, noise_std_priors, data)


# - - - - - - - - Boss Options - - - - - - - -

"""
If an object `cb` of type `BossCallback` is passed to `BossOptions`,
the method shown below will be called before the BO procedure starts
and after every iteration.

```
cb(problem::BossProblem;
    model_fitter::ModelFitter,
    acq_maximizer::AcquisitionMaximizer,
    acquisition::AcquisitionFunction,
    term_cond::TermCond,
    options::BossOptions,
    first::Bool,
)
```

The kwargs `first` is true on the first callback before the BO procedure starts,
and is false on all subsequent callbacks after each iteration.

See `PlotCallback` for an example usage of this feature for plotting.
"""
abstract type BossCallback end

"""
    NoCallback()

Does nothing.
"""
struct NoCallback <: BossCallback end
(::NoCallback)(::BossProblem; kwargs...) = nothing

"""
    BossOptions(; kwargs...)

Stores miscellaneous settings of the BOSS algorithm.

# Keywords
- `info::Bool`: Setting `info=false` silences the BOSS algorithm.
- `debug::Bool`: Set `debug=true` to print stactraces of caught optimization errors.
- `parallel_evals::Symbol`: Possible values: `:serial`, `:parallel`, `:distributed`. Defaults to `:parallel`.
        Determines whether to run multiple objective function evaluations
        within one batch in serial, parallel, or distributed fashion.
        (Only has an effect if batching AM is used.)
- `callback::BossCallback`: If provided, `callback(::BossProblem; kwargs...)`
        will be called before the BO procedure starts and after every iteration.

See also: [`bo!`](@ref)
"""
struct BossOptions{
    CB<:BossCallback
}
    info::Bool
    debug::Bool
    parallel_evals::Symbol
    callback::CB
end
function BossOptions(;
    info = true,
    debug = false,
    parallel_evals = :parallel,
    callback = NoCallback(),
)
    @assert parallel_evals in [:serial, :parallel, :distributed]
    return BossOptions(info, debug, parallel_evals, callback)
end
