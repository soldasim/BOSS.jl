# Types
include("types/include.jl")

# Code
include("utils/include.jl")
include("acquisition.jl")
include("posterior.jl")
include("params_prior.jl")
include("bo.jl")
include("plot.jl")

# Modules
include("acquisitions/include.jl")
include("models/include.jl")
include("data/include.jl")
include("term_conds/include.jl")

# Algorithms
include("model_fitters/include.jl")
include("acquisition_maximizers/include.jl")

# Other
include("deprecated.jl")
