module BOSS

# functions
export bo!, result

# Problem Definition
export OptimizationProblem
export NoFitness, LinFitness, NonlinFitness
export Domain
export ExperimentData

# Surrogate Models
export Parametric, LinModel, NonlinModel
export Nonparametric, GaussianProcess
export Semiparametric

# Model Fitters
export RandomMLE
export SamplingMLE
export OptimizationMLE
export TuringBI

# Acquisition Maximizers
export RandomAM
export GridAM
export OptimizationAM

# Acquisition Functions
export ExpectedImprovement

# Termination Conditions
export IterLimit

# Miscellaneous
export AbstractBounds
export BossOptions
export PlotOptions

include("include.jl")

end # module BOSS
