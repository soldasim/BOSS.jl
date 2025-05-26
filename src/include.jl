# Types
include("types/include.jl")

# Code
include("utils/include.jl")
include("acquisition.jl")
include("posterior.jl")
include("bo.jl")
include("plot.jl")

# Modules
include("acquisitions/include.jl")
include("models/include.jl")
include("term_conds/include.jl")

# Algotithms
include("model_fitters/include.jl")
include("acquisition_maximizers/include.jl")

# Other
include("deprecated.jl")
