
# - - - - - - - - Miscellaneous - - - - - - - -

"""
An abstract type used to differentiate between
`MAP` (Maximum A Posteriori) and `BI` (Bayesian Inference) types.
"""
abstract type ModelFit end
struct MAP <: ModelFit end
struct BI <: ModelFit end


# - - - - - - - - Fitness - - - - - - - -

"""
An abstract type for a fitness function
measuring the quality of an output `y` of the objective function.

Fitness is used by the `AcquisitionFunction` to determine promising points for future evaluations.

All fitness types *should* implement:
- (::CustomFitness)(y::AbstractVector{<:Real}) -> fitness::Real

An exception is the `NoFitness`, which can be used for problem without a well defined fitness.
In such case, an `AcquisitionFunction` which does not depend on `Fitness` must be used.

See also: [`NoFitness`](@ref), [`LinFitness`](@ref), [`NonlinFitness`](@ref), [`AcquisitionFunction`](@ref)
"""
abstract type Fitness end

"""
    NoFitness()

Placeholder for problems with no defined fitness.
    
`BossProblem` defined with `NoFitness` can only be solved with `AcquisitionFunction` not dependent on `Fitness`.
"""
struct NoFitness <: Fitness end


# - - - - - - - - Domain - - - - - - - -

"""
    const AbstractBounds = Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}

Defines box constraints.

Example: `([0, 0], [1, 1]) isa AbstractBounds`
"""
const AbstractBounds = Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}

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


# - - - - - - - - Model (Hyper)Parameters & Priors - - - - - - - -

"""
    const Theta = AbstractVector{<:Real}

Parameters of the parametric model. Is empty in case of a nonparametric model.

Example: `[1., 2., 3.] isa Theta`
"""
const Theta = AbstractVector{<:Real}

"""
    const LengthScales = Union{Nothing, <:AbstractMatrix{<:Real}}

Length scales of the GP as a `x_dim`×`y_dim` matrix, or `nothing` if the model is purely parametric.

Example: `[1.;5.;; 1.;5.;;] isa LengthScales`
"""
const LengthScales = Union{Nothing, <:AbstractMatrix{<:Real}}

"""
    const Amplitudes = Union{Nothing, <:AbstractVector{<:Real}}

Amplitudes of the GP, or `nothing` if the model is purely parametric.

Example: `[1., 5.] isa Amplitudes`
"""
const Amplitudes = Union{Nothing, <:AbstractVector{<:Real}}

"""
    const NoiseStd = AbstractVector{<:Real}

Noise standard deviations of each `y` dimension.

Example: `[0.1, 1.] isa NoiseStd`
"""
const NoiseStd = AbstractVector{<:Real}

"""
    const ModelParams = Tuple{
        <:Theta,
        <:LengthScales,
        <:Amplitudes,
        <:NoiseStd,
    }

Represents all model (hyper)parameters.

Example:
```
params = (nothing, [1.;π;; 1.;π;;], [1., 1.5], [0.1, 1.])
params isa BOSS.ModelParams

θ, λ, α, noise = params
θ isa BOSS.Theta
λ isa BOSS.LengthScales
α isa BOSS.Amplitudes
noise isa BOSS.NoiseStd
```

See: [`Theta`](@ref), [`LengthScales`](@ref), [`Amplitudes`](@ref), [`NoiseStd`](@ref)
"""
const ModelParams = Tuple{
    <:Theta,
    <:LengthScales,
    <:Amplitudes,
    <:NoiseStd,
}

"""
    const ThetaPriors = AbstractVector{<:UnivariateDistribution}

Prior of [`Theta`](@ref).
"""
const ThetaPriors = AbstractVector{<:UnivariateDistribution}

"""
    const LengthScalePriors = Union{Nothing, <:AbstractVector{<:MultivariateDistribution}}

Prior of [`LengthScales`](@ref).
"""
const LengthScalePriors = Union{Nothing, <:AbstractVector{<:MultivariateDistribution}}

"""
    const AmplitudePriors = Union{Nothing, <:AbstractVector{<:UnivariateDistribution}}

Prior of [`Amplitudes`](@ref).
"""
const AmplitudePriors = Union{Nothing, <:AbstractVector{<:UnivariateDistribution}}

"""
    const NoiseStdPriors = AbstractVector{<:UnivariateDistribution}

Prior of [`NoiseStd`](@ref).
"""
const NoiseStdPriors = AbstractVector{<:UnivariateDistribution}

"""
    const ParamPriors = Tuple{
        <:ThetaPriors,
        <:LengthScalePriors,
        <:AmplitudePriors,
        <:NoiseStdPriors,
    }

Represents all (hyper)parameter priors.

See: [`ThetaPriors`](@ref), [`LengthScalePriors`](@ref), [`AmplitudePriors`](@ref), [`NoiseStdPriors`](@ref)
"""
const ParamPriors = Tuple{
    <:ThetaPriors,
    <:LengthScalePriors,
    <:AmplitudePriors,
    <:NoiseStdPriors,
}


# - - - - - - - - Surrogate Model - - - - - - - -

"""
An abstract type for a surrogate model approximating the objective function.

Example usage: `struct CustomModel <: SurrogateModel ... end`

All models *should* implement:
- `make_discrete(model::CustomModel, discrete::AbstractVector{<:Bool}) -> discrete_model::CustomModel`
- `model_posterior(model::CustomModel, data::ExperimentDataMAP) -> (x -> mean, std)`
- `model_loglike(model::CustomModel, data::ExperimentData) -> (::ModelParams -> ::Real)`
- `sample_params(model::CustomModel) -> ::ModelParams`
- `param_priors(model::CustomModel) -> ::ParamPriors`

Models *may* implement:
- `model_posterior_slice(model::CustomModel, data::ExperimentDataMAP, slice::Int) -> (x -> mean, std)`

Model can be designated as "sliceable" by defining `sliceable(::CustomModel) = true`.
A sliceable model *should* additionally implement:
- `model_loglike_slice(model::SliceableModel, data::ExperimentData, slice::Int) -> (::ModelParams -> ::Real)`
- `θ_slice(model::SliceableModel, idx::Int) -> Union{Nothing, UnitRange{<:Int}}`

See also:
[`LinModel`](@ref), [`NonlinModel`](@ref),
[`GaussianProcess`](@ref),
[`Semiparametric`](@ref)
"""
abstract type SurrogateModel end

# Specific implementations of `SurrogateModel` are in '\src\models'.


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
- `params::ModelParams`: Contains MAP model (hyper)parameters.
- `consistent::Bool`: True iff the parameters have been fitted using the current dataset (`X`, `Y`).
        Is set to `consistent = false` after updating the dataset,
        and to `consistent = true` after re-fitting the parameters.

See also: [`ExperimentDataBI`](@ref)
"""
mutable struct ExperimentDataMAP{
    T<:AbstractMatrix{<:Real},
    P<:ModelParams,
} <: ExperimentDataPost{MAP}
    X::T
    Y::T
    params::P
    consistent::Bool
end

"""
Stores the data matrices `X`,`Y` as well as the sampled model parameters and hyperparameters.

# Fields
- `X::AbstractMatrix{<:Real}`: Contains the objective function inputs as columns.
- `Y::AbstractMatrix{<:Real}`: Contains the objective function outputs as columns.
- `params::AbstractVector{<:ModelParams}`: Contains samples of the model (hyper)parameters.
- `consistent::Bool`: True iff the parameters have been fitted using the current dataset (`X`, `Y`).
        Is set to `consistent = false` after updating the dataset,
        and to `consistent = true` after re-fitting the parameters.

See also: [`ExperimentDataMAP`](@ref)
"""
mutable struct ExperimentDataBI{
    T<:AbstractMatrix{<:Real},
    P<:AbstractVector{<:ModelParams},
} <: ExperimentDataPost{BI}
    X::T
    Y::T
    params::P
    consistent::Bool
end


# - - - - - - - - Optimization Problem - - - - - - - -

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
- `fitness::Fitness`: The [`Fitness`](@ref) function.
- `f::Union{Function, Missing}`: The objective blackbox function.
- `domain::Domain`: The [`Domain`](@ref) of `x`.
- `y_max`: The constraints on `y`. (See the definition above.)
- `model::SurrogateModel`: The [`SurrogateModel`](@ref).
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
    data::ExperimentData
end
BossProblem(;
    fitness = NoFitness(),
    f,
    domain,
    model,
    data,
    y_max = fill(Inf, y_dim(data)),
) = BossProblem(fitness, f, domain, y_max, model, data)


# - - - - - - - - Acquisition Function - - - - - - - -

"""
Specifies the acquisition function describing the "quality" of a potential next evaluation point.
Inherit this type to define a custom acquisition function.

Example: `struct CustomAcq <: AcquisitionFunction ... end`

All acquisition functions *should* implement:
`(acquisition::CustomAcq)(problem::BossProblem, options::BossOptions)`

This method should return a function `acq(x::AbstractVector{<:Real}) = val::Real`,
which is maximized to select the next evaluation function of blackbox function in each iteration.

See also: [`ExpectedImprovement`](@ref)
"""
abstract type AcquisitionFunction end

# Specific implementations of `AcquisitionFunction` are in '\src\acquisition'.


# - - - - - - - - Acquisition Maximizer - - - - - - - -

"""
Specifies the library/algorithm used for acquisition function optimization.
Inherit this type to define a custom acquisition maximizer.

Example: `struct CustomAlg <: AcquisitionMaximizer ... end`

Structures derived from this type have to implement the following method:
`maximize_acquisition(acq_maximizer::CustomAlg, acq::AcquisitionFunction, problem::BossProblem, options::BossOptions)`.

This method should return a tuple `(x, val)`.
The returned `x` is the point of the input domain which maximizes the given acquisition function `acq` (as a vector),
or a batch of points (as a column-wise matrix).
The returned `val` is the acquisition value `acq(x)`,
or the values `acq.(eachcol(x))` for each point of the batch,
or `nothing` (depending on the acquisition maximizer implementation).

See also: [`OptimizationAM`](@ref)
"""
abstract type AcquisitionMaximizer end

# Specific implementations of `AcquisitionMaximizer` are in '\src\acquisition_maximizer'.


# - - - - - - - - Model Fitter - - - - - - - -

"""
Specifies the library/algorithm used for model parameter estimation.
Inherit this type to define a custom model-fitting algorithms.

Example: `struct CustomFitter <: ModelFitter{MAP} ... end` or `struct CustomFitter <: ModelFitter{BI} ... end`

Structures derived from this type have to implement the following method:
`estimate_parameters(model_fitter::CustomFitter, problem::BossProblem; info::Bool)`.

This method should return a tuple `(params, val)`.
The returned `params` should be a `ModelParams` (if `CustomAlg <: ModelFitter{MAP}`)
or a `AbstractVector{<:ModelParams}` (if `CustomAlg <: ModelFitter{BI}`).
The returned `val` should be the log likelihood of the parameters (if `CustomAlg <: ModelFitter{MAP}`),
or a vector of log likelihoods of the individual parameter samples (if `CustomAlg <: ModelFitter{BI}`),
or `nothing`.

See also: [`OptimizationMAP`](@ref), [`TuringBI`](@ref)
"""
abstract type ModelFitter{T<:ModelFit} end

# Specific implementations of `ModelFitter` are in '\src\model_fitter'.


# - - - - - - - - Termination Condition - - - - - - - -

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


# - - - - - - - - Callback - - - - - - - -

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


# - - - - - - - - Options - - - - - - - -

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
