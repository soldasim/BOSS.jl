using Distributions
using AbstractGPs

# TODO: refactor types

const AbstractBounds = Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}


# - - - - - - - - Fitness Function - - - - - - - -

"""
An abstract type for a fitness function
measuring the quality of an output `y` of the objective function.

See also: [`BOSS.LinFitness`](@ref), [`BOSS.NonlinFitness`](@ref)
"""
abstract type Fitness end

"""
    LinFitness(coefs::AbstractVector{<:Real})

Used to define a linear fitness function 
measuring the quality of an output `y` of the objective function.

Provides better performance than using the more general `BOSS.NonlinFitness`
as some acquisition functions can be calculated analytically with linear fitness
functions whereas this may not be possible with a nonlinear fitness function.

See also: [`BOSS.NonlinFitness`](@ref)

# Example
A fitness function `f(y) = y[1] + a * y[2] + b * y[3]` can be defined as:
```julia-repl
julia> BOSS.LinFitness([1., a, b])
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

If your fitness function is linear, use `BOSS.LinFitness` instead for better performance.

See also: [`BOSS.LinFitness`](@ref)

# Example
```julia-repl
julia> NonlinFitness(y -> cos(y[1]) + sin(y[2]))
```
"""
struct NonlinFitness <: Fitness
    fitness::Function
end
(f::NonlinFitness)(y) = f.fitness(y)


# - - - - - - - - Surrogate Model - - - - - - - -

"""
An abstract type for a surrogate model approximating the objective function.

See also:
[`BOSS.LinModel`](@ref), [`BOSS.NonlinModel`](@ref),
[`BOSS.Nonparametric`](@ref),
[`BOSS.Semiparametric`](@ref)
"""
abstract type SurrogateModel end

"""
An abstract type for parametric surrogate models.

See also: [`BOSS.LinModel`](@ref), [`BOSS.NonlinModel`](@ref)
"""
abstract type Parametric <: SurrogateModel end

"""
    LinModel(; kwargs...)

A parametric surrogate model linear in its parameters.

This model definition will provide better performance than the more general 'BOSS.NonlinModel' in the future.
This feature is not implemented yet so it is equivalent to using `BOSS.NonlinModel` for now.

The linear model is defined as
    ϕs = lift(x)
    y = [θs[i]' * ϕs[i] for i in 1:m]
where
    x = [x₁, ..., xₙ]
    y = [y₁, ..., yₘ]
    θs = [θ₁, ..., θₘ], θᵢ = [θᵢ₁, ..., θᵢₚ]
    ϕs = [ϕ₁, ..., ϕₘ], ϕᵢ = [ϕᵢ₁, ..., ϕᵢₚ]
     n, m, p ∈ R.

# Keywords
- `lift::Function`: Define the `lift` function according to the definition above.
- `param_priors::AbstractVector{<:UnivariateDistribution}`: The prior distributions for
        the parameters `[θ₁₁, ..., θ₁ₚ, ..., θₘ₁, ..., θₘₚ]` according to the definition above.
- `discrete::Union{Nothing, <:AbstractVector{<:Bool}}`: Describes which dimensions are discrete.
        Typically, the `discrete` kwarg can be ignored by the end-user as it will be updated
        according to the `Domain` of the `OptimizationProblem` during BOSS initialization.
"""
struct LinModel{
    P<:AbstractVector{<:UnivariateDistribution},
    D<:Union{Nothing, <:AbstractVector{<:Bool}},
} <: Parametric
    lift::Function
    param_priors::P
    discrete::D
end
LinModel(;
    lift,
    param_priors,
    discrete=nothing,
) = LinModel(lift, param_priors, discrete)

"""
    NonlinModel(; kwargs...)

A parametric surrogate model.

If your model is linear, you can use `BOSS.LinModel` which will provide better performance in the future. (Not yet implemented.)

Define the model as `y = predict(x, θ)` where `θ` are the model parameters.

# Keywords
- `predict::Function`: The `predict` function according to the definition above.
- `param_priors::AbstractVector{<:UnivariateDistribution}`: The prior distributions for the model parameters.
- `discrete::Union{Nothing, <:AbstractVector{<:Bool}}`: Describes which dimensions are discrete.
        Typically, the `discrete` kwarg can be ignored by the end-user as it will be updated
        according to the `Domain` of the `OptimizationProblem` during BOSS initialization.
"""
struct NonlinModel{
    P<:AbstractVector{<:UnivariateDistribution},
    D<:Union{Nothing, <:AbstractVector{<:Bool}},
} <: Parametric
    predict::Function
    param_priors::P
    discrete::D
end
NonlinModel(;
    predict,
    param_priors,
    discrete=nothing,
) = NonlinModel(predict, param_priors, discrete)

(m::ZeroMean)(x::AbstractVector{<:Real}) = zeros(eltype(x), m.y_dim)

"""
    Nonparametric(; kwargs...)

A Gaussian Process surrogate model.

# Keywords
- `mean::Union{Nothing, Function}`: Used as the mean function for the GP.
        Defaults to `nothing` equivalent to `x -> 0.`.
- `kernel::Kernel`: The kernel used in the GP. Defaults to the `Matern52Kernel()`.
- `length_scale_priors::AbstractVector{<:MultivariateDistribution}`: The prior distributions
        for the length scales of the GP. The `length_scale_priors` should be a vector
        of `y_dim` `x_dim`-variate distributions where `x_dim` and `y_dim` are
        the dimensions of the input and output of the model respectively.
"""
struct Nonparametric{
    M<:Union{Nothing, Function},
    P<:AbstractVector{<:MultivariateDistribution},
} <: SurrogateModel
    mean::M
    kernel::Kernel
    length_scale_priors::P
end
Nonparametric(;
    mean=nothing,
    kernel=Matern52Kernel(),
    length_scale_priors,
) = Nonparametric(mean, kernel, length_scale_priors)

"""
    Semiparametric(; kwargs...)

A semiparametric surrogate model (a combination of a parametric model and Gaussian Process).

The parametric model is used as the mean of the Gaussian Process and all parameters
and hyperparameters are estimated simultaneously.

# Keywords
- `parametric::Parametric`: The parametric model used as the GP mean function.
- `nonparametric::Nonparametric{Nothing}`: The outer GP model without mean.
"""
struct Semiparametric{
    P<:Parametric,
    N<:Nonparametric{Nothing},
} <: SurrogateModel
    parametric::P
    nonparametric::N
end

# - - - - - - - - Model-Fit Algorithms - - - - - - - -

"""
An abstract type used to differentiate between
MLE (Maximum Likelihood Estimation) optimizers and BI (Bayesian Inference) samplers.
"""
abstract type ModelFit end
struct MLE <: ModelFit end
struct BI <: ModelFit end

"""
Specifies the library/algorithm used for model parameter estimation.
Inherit this type to define a custom model-fitting algorithms.

Example: `struct CustomFitter <: ModelFitter{MLE} ... end` or `struct CustomFitter <: ModelFitter{BI} ... end`

Structures derived from this type have to implement the following method:
`estimate_parameters(model_fitter::CustomFitter, problem::OptimizationProblem; info::Bool)`.

This method should return a named tuple `(θ = ..., length_scales = ..., noise_vars = ...)`
with either MLE model parameters (if `CustomAlg <: ModelFitter{MLE}`)
or model parameter samples (if `CustomAlg <: ModelFitter{BI}`).

See also: [`BOSS.OptimMLE`](@ref), [`BOSS.TuringBI`](@ref)
"""
abstract type ModelFitter{T<:ModelFit} end

# Specific implementations of `ModelFitter` are in '\src\model_fitter'.


# - - - - - - - - Acquisition Maximization - - - - - - - -

"""
Specifies the library/algorithm used for acquisition function optimization.
Inherit this type to define a custom acquisition maximizer.

Example: `struct CustomAlg <: AcquisitionMaximizer ... end`

Structures derived from this type have to implement the following method:
`maximize_acquisition(acq_maximizer::CustomAlg, problem::OptimizationProblem, acq::Function; info::Bool)`
This method should return the point of the input domain which maximizes the given acquisition function `acq`.

See also: [`BOSS.OptimMaximizer`](@ref)
"""
abstract type AcquisitionMaximizer end

# Specific implementations of `AcquisitionMaximizer` are in '\src\acquisition_maximizer'.


# - - - - - - - - Acquisition Functions - - - - - - - -

"""
Specifies the acquisition function describing the "quality" of a potential next evaluation point.
Inherit this type to define a custom acquisition function.

Example: `struct CustomAcq <: AcquisitionFunction ... end`

Structures derived from this type have to implement the following method:
`(acquisition::CustomAcq)(problem::OptimizationProblem, options::BossOptions)`

This method should return a function `acq(x::AbstractVector{<:Real}) = val::Real`,
which is maximized to select the next evaluation function of blackbox function in each iteration.

See also: [`BOSS.ExpectedImprovement`](@ref)
"""
abstract type AcquisitionFunction end

# Specific implementations of `AcquisitionFunction` are in '\src\acquisition'.


# - - - - - - - - Termination Conditions - - - - - - - -

"""
Specifies the termination condition of the whole BOSS algorithm.
Inherit this type to define a custom termination condition.

Example: `struct CustomCond <: TermCond ... end`

Structures derived from this type have to implement the following method:
`(cond::CustomCond)(problem::OptimizationProblem) where {CustomCond <: TermCond}`

This method should return true to keep the optimization running
and return false once the optimization is to be terminated.

See also: [`BOSS.IterLimit`](@ref)
"""
abstract type TermCond end

# Specific implementations of `TermCond` are in '\src\term_cond.jl'.


# - - - - - - - - Data - - - - - - - -

"""
Stores all the data collected during the optimization
as well as the parameters and hyperparameters of the model.

See also: [`BOSS.ExperimentDataPrior`](@ref), [`BOSS.ExperimentDataPost`](@ref)
"""
abstract type ExperimentData end

Base.length(data::ExperimentData) = size(data.X)[2]
Base.isempty(data::ExperimentData) = isempty(data.X)

"""
Stores the initial data.

# Fields
- `X::AbstractMatrix{<:Real}`: Contains the objective function inputs as columns.
- `Y::AbstractMatrix{<:Real}`: Contains the objective function outputs as columns.

See also: [`BOSS.ExperimentDataPost`](@ref)
"""
mutable struct ExperimentDataPrior{
    T<:AbstractMatrix{<:Real},
} <: ExperimentData
    X::T
    Y::T
end

# TODO: Implement an options to pass empty data to BOSS and let it automatically sample few initial samples.
# empty_data(x_dim::Int, y_dim::Int, type::Type=Float64) = ExperimentDataPrior(Array{type}(undef, x_dim, 0), Array{type}(undef, y_dim, 0))

"""
Stores the fitted/samples model parameters in addition to the data matrices `X`,`Y`.

See also: [`BOSS.ExperimentDataPrior`](@ref), [`BOSS.ExperimentDataMLE`](@ref), [`BOSS.ExperimentDataBI`](@ref)
"""
abstract type ExperimentDataPost{T<:ModelFit} <: ExperimentData end

"""
Stores the data matrices `X`,`Y` as well as the optimized model parameters and hyperparameters.

# Fields
- `X::AbstractMatrix{<:Real}`: Contains the objective function inputs as columns.
- `Y::AbstractMatrix{<:Real}`: Contains the objective function outputs as columns.
- `θ::Union{Nothing, <:AbstractVector{<:Real}}`: Contains the MLE parameters
        of the parametric model (or nothing if the model is nonparametric).
- `length_scales::Union{Nothing, <:AbstractMatrix{<:Real}}`: Contains the MLE length scales
        of the GP as a `x_dim`×`y_dim` matrix (or nothing if the model is parametric).
- `noise_vars::AbstractVector{<:Real}`: The MLE noise variances of each `y` dimension.

See also: [`BOSS.ExperimentDataBI`](@ref)
"""
mutable struct ExperimentDataMLE{
    T<:AbstractMatrix{<:Real},
    P<:Union{Nothing, <:AbstractVector{<:Real}},
    L<:Union{Nothing, <:AbstractMatrix{<:Real}},
    N<:AbstractVector{<:Real},
} <: ExperimentDataPost{MLE}
    X::T
    Y::T
    θ::P
    length_scales::L
    noise_vars::N
end

"""
Stores the data matrices `X`,`Y` as well as the sampled model parameters and hyperparameters.

# Fields
- `X::AbstractMatrix{<:Real}`: Contains the objective function inputs as columns.
- `Y::AbstractMatrix{<:Real}`: Contains the objective function outputs as columns.
- `θ::Union{Nothing, <:AbstractMatrix{<:Real}}`: Samples of parameters of the parametric model
        stored column-wise in a matrix (or nothing if the model is nonparametric).
- `length_scales::Union{Nothing, <:AbstractVector{<:AbstractMatrix{<:Real}}}`: Samples
        of the length scales of the GPs as stored column-wise in a matrix for each `y` dimension
        (or nothing if the model is parametric).
- `noise_vars::AbstractMatrix{<:Real}`: Samples of the noise variances of each `y` dimension
        stored column-wise in a matrix.

See also: [`BOSS.ExperimentDataBI`](@ref)
"""
mutable struct ExperimentDataBI{
    T<:AbstractMatrix{<:Real},
    P<:Union{Nothing, <:AbstractMatrix{<:Real}},
    L<:Union{Nothing, <:AbstractVector{<:AbstractMatrix{<:Real}}},
    N<:AbstractMatrix{<:Real},
} <: ExperimentDataPost{BI}
    X::T
    Y::T
    θ::P
    length_scales::L
    noise_vars::N
end


# - - - - - - - - Optimization Problem - - - - - - - -

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
        (See [`BOSS.AcquisitionMaximizer`](@ref).)
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
    discrete=fill(false, length(first(bounds))),
    cons=nothing,
)
    @assert length(bounds[1]) == length(bounds[2]) == length(discrete)
    return Domain(bounds, discrete, cons)
end

"""
    OptimizationProblem(; kwargs...)

Defines the whole optimization problem for the BOSS algorithm.

# Problem Definition

    There is some (noisy) blackbox function `y = f(x) = f_true(x) + ϵ` where `ϵ ~ Normal`.

    We have some surrogate model `y = model(x) ≈ f_true(x)`
    describing our knowledge (or lack of it) about the blackbox function.

    We wish to find `x ∈ domain` such that `fitness(f(x))` is maximized
    while satisfying the constraints `f(x) <= y_max`.

# Keywords
- `fitness::Fitness`: The fitness function. See [`BOSS.Fitness`](@ref).
- `f::Union{Function, Missing}`: The objective blackbox function.
- `domain::Domain`: The domain of `x`. See [`BOSS.Domain`](@ref).
- `y_max`: The constraints on `y`. (See the definition above.)
- `model::SurrogateModel`: See [`BOSS.SurrogateModel`](@ref).
- `noise_var_priors::AbstractVector{<:UnivariateDistribution}`: The prior distributions
        of the noise variances of each `y` dimension.
- `data::ExperimentData`: The initial data of objective function evaluations.
        See [`BOSS.ExperimentDataPrior`].

See also: [`BOSS.boss!`](@ref)
"""
mutable struct OptimizationProblem{
    F<:Union{Function, Missing},
}
    fitness::Fitness
    f::F
    domain::Domain
    y_max::AbstractVector{<:Real}
    model::SurrogateModel
    noise_var_priors::AbstractVector{<:UnivariateDistribution}
    data::ExperimentData
end
OptimizationProblem(;
    fitness,
    f,
    domain,
    model,
    noise_var_priors,
    y_max=fill(Inf, length(noise_var_priors)),
    data,
) = OptimizationProblem(fitness, f, domain, y_max, model, noise_var_priors, data)


# - - - - - - - - Boss Options - - - - - - - -

"""
    PlotOptions(Plots; kwargs...)

If `PlotOptions` is passed to `boss!`, the state of the optimization problem
is plotted in each iteration. Only works with one-dimensional `x` domains
but supports multi-dimensional `y`.

# Arguments
- `Plots::Module`: Evaluate `using Plots` and pass the `Plots` module to `PlotsOptions`.

# Keywords
- `f_true::Union{Nothing, Function}`: The true objective function to be plotted.
- `points::Int`: The number of points in each plotted function.
- `xaxis::Symbol`: Used to change the x axis scale (`:identity`, `:log`).
- `yaxis::Symbol`: Used to change the y axis scale (`:identity`, `:log`).
- `title::String`: The plot title.
"""
struct PlotOptions{
    F<:Union{Nothing, Function},
    A<:Union{Nothing, Function},
    O<:Union{Nothing, AbstractVector{<:Real}},
}
    Plots::Module
    f_true::F
    acquisition::A
    acq_opt::O
    points::Int
    xaxis::Symbol
    yaxis::Symbol
    title::String
end
PlotOptions(Plots::Module;
    f_true=nothing,
    acquisition=nothing,
    acq_opt=nothing,
    points=200,
    xaxis=:identity,
    yaxis=:identity,
    title="BOSS optimization problem",
) = PlotOptions(Plots, f_true, acquisition, acq_opt, points, xaxis, yaxis, title)


"""
    BossOptions(; kwargs...)

Stores miscellaneous settings of the BOSS algorithm.

# Keywords
- `info::Bool`: Setting `info=false` silences the BOSS algorithm.
- `debug::Bool`: Set `debug=true` to print stactraces of caught optimization errors.
- `plot_options::PlotOptions`: If `plot_options` is provided, BOSS will print the state
        of the optimization problem in each iteration. See [`BOSS.PlotOptions`](@ref).

See also: [`BOSS.boss!`](@ref)
"""
struct BossOptions{
    P<:Union{Nothing, PlotOptions},
}
    info::Bool
    debug::Bool
    plot_options::P
end
BossOptions(;
    info=true,
    debug=false,
    plot_options=nothing,
) = BossOptions(info, debug, plot_options)
