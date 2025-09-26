module BOSS

# functions
export bo!
export estimate_parameters!, maximize_acquisition, eval_objective!
export update_parameters!, augment_dataset!
export construct_acquisition
export model_posterior, model_posterior_slice
export mean, std, var, cov
export mean_and_std, mean_and_var, mean_and_cov
export average_mean

# utils
export x_dim, y_dim, cons_dim, data_count, is_consistent
export get_fitness, get_params
export result
export calc_inverse_gamma, TruncatedMvNormal

# Problem Definition
export BossProblem
export Domain, AbstractBounds

# Acquisition Functions
export AcquisitionFunction
export Fitness, LinFitness, NonlinFitness
export ExpectedImprovement

# Surrogate Models
export SurrogateModel, ModelParams, AbstractModelPosterior, ModelPosterior, ModelPosteriorSlice
export Parametric, LinearModel, NonlinearModel, ParametricParams, ParametricPosterior
export Nonparametric, GaussianProcess, GaussianProcessParams, GaussianProcessPosterior
export Semiparametric, SemiparametricParams
export NonstationaryGP, NonstationaryGPParams, ParametrizedGP, ParametrizedGPParams

# Parameters
export FittedParams, UniFittedParams, MultiFittedParams
export FixedParams, RandomParams, MAPParams, BIParams

# Experiment Data
export ExperimentData, SimpleData, NormalizedData

# Model Fitters
export ModelFitter
export RandomFitter
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
export GivenPointAM, GivenSequenceAM

# Miscellaneous
export BossOptions
export TermCond, IterLimit, DataLimit, NoLimit
export BossCallback, NoCallback

# Other
export PlotCallback

# Imports
using Random
using Distributions
import Distributions: mean, std, var, cov
import Distributions: mean_and_var, mean_and_std, mean_and_cov
using LinearAlgebra
using AbstractGPs
using LatinHypercubeSampling
using Optimization
using InteractiveUtils
using Distributed
using Bijectors
using InverseFunctions

include("include.jl")

end # module BOSS
