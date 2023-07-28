using AbstractGPs

const AbstractBounds = Tuple{<:AbstractArray{<:Real}, <:AbstractArray{<:Real}}


# - - - - - - - - Fitness Function - - - - - - - -

"""
An abstract type for a fitness function
measuring the quality of an output `y` of the objective function.

See also: [`BOSS.LinFitness`](@ref), [`BOSS.NonlinFitness`](@ref)
"""
abstract type Fitness end

"""
Used to define a linear fitness function 
measuring the quality of an output `y` of the objective function.

Provides better performance than using the more general `BOSS.NonlinFitness`.

See also: [`BOSS.NonlinFitness`](@ref)

# Example
A fitness function `f(y) = y[1] + a * y[2] + b * y[3]` can be defined as:
```julia-repl
julia> BOSS.LinFitness([1., a, b])
```
"""
struct LinFitness{
    C<:AbstractArray{<:Real},
} <: Fitness
    coefs::C
end
(f::LinFitness)(y) = f.coefs' * y

"""
Used to define a fitness function
measuring the quality of an output `y` of the objective function.

If your fitness function is linear, use `BOSS.LinFitness` for better performance.

See also: [`BOSS.LinFitness`](@ref)

# Example
```julia-repl
julia> NonlinFitness(y -> cos(y[1]) + sin(y[2]))
```
"""
struct NonlinFitness{
    F<:Base.Callable,
} <: Fitness
    fitness::F
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
Used to define a parametric surrogate model linear in its parameters.

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

Define the `lift` function according to the model definition above.
Provide the model parameter priors in the `param_priors` field as an array of distributions.
"""
struct LinModel{
    L<:Base.Callable,
    D<:AbstractArray{<:Any},
} <: Parametric
    lift::L
    param_priors::D
end
LinModel(;
    lift,
    param_priors,
) = LinModel(lift, param_priors)

"""
Used to define a parametric surrogate model.

If your model is linear, you can use `BOSS.LinModel` which will provide better performance in the future. (Not yet implemented.)

Define the `predict` funtion as `y = predict(x,θ)` where `θ` are the model parameters.
Provide the model parameter priors in the `param_priors` field as an array of distributions.
"""
struct NonlinModel{
    P<:Base.Callable,
    D<:AbstractArray,
} <: Parametric
    predict::P
    param_priors::D
end
NonlinModel(;
    predict,
    param_priors,
) = NonlinModel(predict, param_priors)

"""
Used to define a nonparametric surrogate model (Gaussian Process).

The `mean` function is used as the mean of the GP.
Zero-mean is used if `mean` is nothing.

You can select a custom kernel via the `kernel` field.

The length scale priors of the GP have to be provided
as an array of `y_dim` multivariate distributions of dimension `x_dim`
where `x_dim` and `y_dim` are the dimensions of the input and output spaces respectively.

Only `MvLogNormal` length scale priors are supported with the `Turing.NUTS` sampler for now.
"""
struct Nonparametric{
    M<:Union{Nothing, Base.Callable},
    D<:AbstractArray,
} <: SurrogateModel
    mean::M
    kernel::Kernel
    length_scale_priors::D
end
Nonparametric(;
    mean=nothing,
    kernel=Matern52Kernel(),
    length_scale_priors,
) = Nonparametric(mean, kernel, length_scale_priors)

"""
Used to define a semiparametric surrogate model (a combination of a parametric model and GP).

The parametric model is used as the mean of the Gaussian Process.
The provided nonparametric mustn't have mean. (Its `mean` field must be set to `nothing`.)
"""
struct Semiparametric{
    P<:Parametric,
    N<:Nonparametric,
} <: SurrogateModel
    parametric::P
    nonparametric::N

    function Semiparametric(p::Parametric, n::Nonparametric)
        @assert isnothing(n.mean)
        new{typeof(p), typeof(n)}(p, n)
    end
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

Additionally, if the custom algorithm is of type `ModelFitter{BI}`, it has to implement the method
`sample_count(::CustomAlg)` giving the number of parameter samples returned from `estimate_parameters`.

See '\\src\\algorithms' for concrete implementations of `ModelFitter`.

See also: [`BOSS.OptimMLE`](@ref), [`BOSS.TuringBI`](@ref)
"""
abstract type ModelFitter{T<:ModelFit} end

# Specific implementations of `ModelFitter` are in '\src\algorithms'.


# - - - - - - - - EI Maximization - - - - - - - -

"""
Specifies the library/algorithm used for acquisition function optimization.

Extend this type to define a custom acquisition maximizer.

Example: `struct CustomAlg <: AcquisitionMaximizer ... end`

Structures derived from this type have to implement the following method:
`maximize_acquisition(acq_maximizer::CustomAlg, problem::OptimizationProblem, acq::Base.Callable; info::Bool)`
This method should return the point of the input domain which maximizes the given acquisition function `acq`.

See '\\src\\algorithms' for concrete implementations of `AcquisitionMaximizer`.

See also: [`BOSS.OptimMaximizer`](@ref)
"""
abstract type AcquisitionMaximizer end

# Specific implementations of `AcquisitionMaximizer` are in '\src\algorithms'.


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
The structures deriving this type contain all the data collected during the optimization
as well as the parameters and hyperparameters of the model.

See also: [`BOSS.ExperimentDataPrior`](@ref), [`BOSS.ExperimentDataPost`](@ref)
"""
abstract type ExperimentData{NUM<:Real} end

Base.length(data::ExperimentData) = size(data.X)[2]
Base.isempty(data::ExperimentData) = isempty(data.X)

"""
Contains the initial data matrices `X`,`Y`.

The data points are stored in the columns of these matrices.

See also: [`BOSS.ExperimentDataPost`](@ref)
"""
mutable struct ExperimentDataPrior{
    NUM<:Real,
    T<:AbstractMatrix{NUM},
} <: ExperimentData{NUM}
    X::T
    Y::T
end

# TODO: Implement an options to pass empty data to BOSS and let it automatically sample few initial samples.
# empty_data(x_dim::Int, y_dim::Int, type::Type=Float64) = ExperimentDataPrior(Array{type}(undef, x_dim, 0), Array{type}(undef, y_dim, 0))

"""
An abstract type for data which contain the fitted model parameters
in addition to the data matrices `X`,`Y`.

See also: [`BOSS.ExperimentDataPrior`](@ref), [`BOSS.ExperimentDataMLE`](@ref), [`BOSS.ExperimentDataBI`](@ref)
"""
abstract type ExperimentDataPost{T<:ModelFit, NUM<:Real} <: ExperimentData{NUM} end

"""
Contains the data matrices `X`,`Y` as well as the optimized model parameters and hyperparameters.

The data points are stored in the columns of the `X`,`Y` matrices.

The field `θ` contains the MLE parameters of the parametric model
(or nothing if the model is nonparametric).

The field `length_scales` contains the MLE length scales of the GP
(or nothing if the model is parametric) as a `x_dim`×`y_dim` matrix.

The field `noise_vars` contains the MLE noise variances for each `y` dimension.

See also: [`BOSS.ExperimentDataBI`](@ref)
"""
mutable struct ExperimentDataMLE{
    NUM<:Real,
    T<:AbstractMatrix{NUM},
    P<:Union{Nothing, <:AbstractArray{NUM}},
    L<:Union{Nothing, <:AbstractMatrix{NUM}},
    N<:Union{Nothing, <:AbstractArray{NUM}},
} <: ExperimentDataPost{MLE, NUM}
    X::T
    Y::T
    θ::P
    length_scales::L
    noise_vars::N
end

"""
Contains the data matrices `X`,`Y` as well as the sampled model parameters and hyperparameters.

The data points are stored in the columns of the `X`,`Y` matrices.

The field `θ` contains the samples of the parameters of the parametric model
as a matrix with column-wise stored samples (or nothing if the model is nonparametric).

The field `length_scales` contains samples of the GP length scales
as an array of matrices (or nothing if the model is parametric).

The field `noise_vars` contains samples of noise variances for each `y` dimension
as a matrix with column-wise stored samples.

See also: [`BOSS.ExperimentDataBI`](@ref)
"""
mutable struct ExperimentDataBI{
    NUM<:Real,
    T<:AbstractMatrix{NUM},
    P<:Union{Nothing, <:AbstractMatrix{NUM}},
    L<:Union{Nothing, <:AbstractArray{<:AbstractMatrix{NUM}}},
    N<:Union{Nothing, <:AbstractMatrix{NUM}},
} <: ExperimentDataPost{BI, NUM}
    X::T
    Y::T
    θ::P
    length_scales::L
    noise_vars::N
end


# - - - - - - - - Optimization Problem - - - - - - - -

"""
This structure defines the whole optimization problem for the BOSS algorithm.

The problem is defined as follows:

There is some (possibly noisy) blackbox function `y = f(x) = f_true(x) + ϵ` where `ϵ ~ Normal`.

We have some surrogate model `y = model(x) ≈ f_true(x)`
describing our limited knowledge about the blackbox function.

We wish to find `x ∈ domain` such that `fitness(f(x))` is maximized
while satisfying the constraints `f(x) < cons`.

The `noise_var_priors` describe the prior distribution over the noise variance of each `y` dimension.
They are defined as an array of `y_dim` distributions where `y_dim` is the dimension of the output space.

See also: [`BOSS.boss!`](@ref), [`BOSS.Fitness`](@ref), [`BOSS.Surrogate`](@ref), [`BOSS.Domain`](@ref)
"""
mutable struct OptimizationProblem{
    NUM<:Real,
    Q<:Fitness,
    F<:Base.Callable,
    C<:AbstractArray{NUM},
    D<:AbstractBounds,
    I<:AbstractArray{<:Bool},
    M<:SurrogateModel,
    N<:AbstractArray,
}
    fitness::Q
    f::F
    cons::C
    domain::D
    discrete::I
    model::M
    noise_var_priors::N
    data::ExperimentData
end
OptimizationProblem(;
    fitness,
    f,
    cons,
    domain,
    discrete,
    model,
    noise_var_priors,
    data,
) = OptimizationProblem(fitness, f, cons, domain, discrete, model, noise_var_priors, data)


# - - - - - - - - Boss Options - - - - - - - -

"""
Is used to pass the `Plots` module to the plotting function `BOSS.plot_problem`
as well as pass any additional information to be plotted and to set other plot hyperparameters.

The `f_true` field can be used to add the true objective function to the plot.
The `acquisition` field can be used to add the acquisition function to the plot.
The `acq_opt` field can be used to add the acquisition maximization result to the plot.

The `points` hyperparameter can be used to change the resolution of the plots.
The `xaxis` and `yaxis` hyperparameters can be used to change the axis scales.
The `title` hyperparameter can be used to change the plot title.

See also: [`BOSS.OptimizationProblem`](@ref), [`BOSS.plot_problem`](@ref)
"""
struct PlotOptions{
    F<:Union{Nothing, Function},
    A<:Union{Nothing, Function},
    O<:Union{Nothing, AbstractArray{<:Real}},
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
Used to pass hyperparameters and miscellaneous settings to the BOSS algorithm.

The `info` can be set to false to silence the algorithm.

The `ϵ_samples` hyperparameter controls how many samples are used to approximate
the posterior predictions of the model. Note that this hyperparameter only has
an effect if a MLE optimizer is used to fit the model parameters.
The number of `ϵ_samples` is matched to the number of samples drawn
by the BI sampler if a BI sampler is used to sample the model parameters.

The `BOSS.PlotOptions` structure can be passed in the `plot_options` field
to turn on plotting and modify its settings.

See also: [`BOSS.boss!`](@ref), [`BOSS.PlotOptions`](@ref)
"""
struct BossOptions{
    P<:Union{Nothing, PlotOptions},
}
    info::Bool
    ϵ_samples::Int  # only for MLE, in case of BI ϵ_samples == sample_count(ModelFitterBI)
    plot_options::P
end
BossOptions(;
    info=true,
    ϵ_samples=200,
    plot_options=nothing,
) = BossOptions(info, ϵ_samples, plot_options)
