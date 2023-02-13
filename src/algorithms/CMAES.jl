using Evolutionary

struct CMAESMaximizer{O} <: AcquisitionMaximizer where {
    O<:Evolutionary.Options,
}
    options::O
    multistart::Int
    parallel::Bool
end

# TODO implement








# UNUSED

# optim_CMAES(f, start; kwargs...) = optim_CMAES(f, Evolutionary.NoConstraints(), start; kwargs...)
# optim_CMAES(f, bounds::Tuple, start; kwargs...) = optim_CMAES(f, Evolutionary.BoxConstraints(bounds...), start; kwargs...)

# function optim_CMAES(f, start, constraints::Evolutionary.AbstractConstraints; options=Evolutionary.Options(; Evolutionary.default_options(CMAES())...), info=false)
#     res = Evolutionary.optimize(x->-f(x), constraints, start, CMAES(), options)
#     info && (Evolutionary.iterations(res) == options.iterations) && @warn "Maximum iterations reached while optimizing!"
#     return res.minimizer, f(res.minimizer)
# end
