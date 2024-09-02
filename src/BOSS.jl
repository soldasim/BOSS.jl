module BOSS

# functions
export bo!
export augment_dataset!
export model_posterior, model_posterior_slice, average_posterior
export result

# Problem Definition
export BossProblem
export Fitness, NoFitness, LinFitness, NonlinFitness
export Domain

# Experiment Data
export ExperimentData
export ExperimentDataPrior, ExperimentDataPost
export ExperimentDataMAP, ExperimentDataBI

# Surrogate Models
export SurrogateModel
export Parametric, LinModel, NonlinModel
export Nonparametric, GaussianProcess
export Semiparametric

# Model Fitters
export ModelFit, MAP, BI
export ModelFitter
export RandomMAP
export SamplingMAP
export OptimizationMAP
export SampleOptMAP
export TuringBI

# Acquisition Maximizers
export AcquisitionMaximizer
export SequentialBatchAM
export RandomAM
export GridAM
export SamplingAM
export OptimizationAM
export SampleOptAM

# Acquisition Functions
export AcquisitionFunction
export ExpectedImprovement

# Termination Conditions
export TermCond
export IterLimit

# Miscellaneous
export AbstractBounds
export BossOptions

# Callbacks
export BossCallback
export NoCallback, PlotCallback

using Random
using Distributions
using AbstractGPs
using LatinHypercubeSampling
using Optimization
using Turing
using InteractiveUtils
using Distributed

include("include.jl")

end # module BOSS
