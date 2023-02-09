
struct NLoptMaximizer{O} <: AcquisitionMaximizer where {
    O<:NLopt.Opt,
}
    options::O
    multistart::Int
    parallel::Bool
end

# TODO implement








# UNUSED

function optim_NLopt(f, start; optimizer=nothing, info=false)
    isnothing(optimizer) && (optimizer = Opt(:LD_MMA, length(start)))
    
    function f_nlopt(x::Vector, grad::Vector)
        if length(grad) > 0
            grad .= ForwardDiff.gradient(f, x)
        end
        return f(x)
    end
    
    optimizer.max_objective = f_nlopt
    val, arg, ret = NLopt.optimize(optimizer, start)
    info && check_nlopt_convergence(ret)
    return arg, val
end

function check_nlopt_convergence(ret)
    if ret != :XTOL_REACHED
        @warn "Optimization terminated with return value `$ret`."
    end
end

# TODO remove
# # TODO: Check if parallelization is safe.
# function opt_acq_NLopt(acq, domain; x_dim::Int, multistart=1, discrete_dims=nothing, optimizer=nothing, parallel, info=true, debug=false)
#     info && parallel && @warn "TODO: Check that parallelization is safe with NLopt!"
    
#     isnothing(optimizer) && (optimizer = Opt(:LD_MMA, length(start)))
#     optimizer.lower_bounds = domain[1]
#     optimizer.upper_bounds = domain[2]
    
#     starts = generate_starts(domain, multistart; x_dim)
#     arg, val = optim_NLopt_multistart(acq, starts; optimizer, parallel, info)

#     isnothing(discrete_dims) || (arg = discrete_round(discrete_dims)(arg))
#     return arg, val
# end
