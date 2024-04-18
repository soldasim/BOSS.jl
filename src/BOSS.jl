module BOSS

# functions
export bo!, result

# Problem Definition
export BossProblem
export Fitness, NoFitness, LinFitness, NonlinFitness
export Domain

# Experiment Data
export ExperimentData
export ExperimentDataPrior, ExperimentDataPost
export ExperimentDataMLE, ExperimentDataBI

# Surrogate Models
export SurrogateModel
export Parametric, LinModel, NonlinModel
export Nonparametric, GaussianProcess
export Semiparametric

# Model Fitters
export ModelFit, MLE, BI
export ModelFitter
export RandomMLE
export SamplingMLE
export OptimizationMLE
export TuringBI

# Acquisition Maximizers
export AcquisitionMaximizer
export SequentialBatchAM
export RandomAM
export GridAM
export OptimizationAM

# Acquisition Functions
export AcquisitionFunction
export ExpectedImprovement

# Termination Conditions
export TermCond
export IterLimit

# Miscellaneous
export AbstractBounds
export BossOptions
export PlotOptions

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
