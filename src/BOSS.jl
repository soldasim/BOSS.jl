module BOSS

# functions
export bo!
export augment_dataset!
export model_posterior, model_posterior_slice, average_posterior
export result

# Problem Definition
export BossProblem
export Fitness, NoFitness, LinFitness, NonlinFitness
export Domain, AbstractBounds

# Experiment Data
export ExperimentData
export ExperimentDataPrior, ExperimentDataPost
export ExperimentDataMAP, ExperimentDataBI

# Surrogate Models
export SurrogateModel
export Parametric, LinModel, NonlinModel
export Nonparametric, GaussianProcess
export Semiparametric

# Acquisition Functions
export AcquisitionFunction
export ExpectedImprovement

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

# Miscellaneous
export BossOptions
export TermCond, IterLimit, NoLimit
export BossCallback, NoCallback

# Other
export PlotCallback

using Random
using Distributions
using LinearAlgebra
using AbstractGPs
using LatinHypercubeSampling
using Optimization
using Turing
using InteractiveUtils
using Distributed

include("include.jl")

end # module BOSS
