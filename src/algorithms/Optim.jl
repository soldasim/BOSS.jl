using Optim
using Threads


# - - - - - - - - Model Parameter MLE - - - - - - - -

struct OptimMLE{
    A<:Optim.Fminbox,
    O<:Optim.Options,
} <: ModelFitter{MLE}
    algorithm::A
    options::O
    multistart::Int
    parallel::Bool
end

function estimate_parameters(opt::OptimMLE, problem::OptimizationProblem; info::Bool)
    loglike = model_loglike(problem.model, problem.noise_var_prior, problem.data)
    θ, length_scales, noise_vars = opt_params(loglike, opt, problem.model, problem.noise_var_prior; x_dim=x_dim(problem), y_dim=y_dim(problem), info)
    return (θ=θ, length_scales=length_scales, noise_vars=noise_vars)
end

function opt_params(loglike::Base.Callable, optim::OptimMLE, model::Parametric, noise_var_prior; x_dim::Int, y_dim::Int, info::Bool)
    split(p) = p[1:y_dim], p[y_dim+1:end]
    function ll(p)
        noise_vars, θ = split(p)
        loglike(θ, noise_vars)
    end

    starts = reduce(hcat, [vcat(
        rand.(noise_priors),
        rand.(model.param_priors),
    ) for _ in 1:optim.multistart])
    bounds = (
        vcat(minimum(noise_var_prior), minimum.(model.param_priors)),
        vcat(maximum(noise_var_prior), maximum.(model.param_priors))
    )

    optimize(start) = optim_maximize(ll, bounds, optim.algorithm, optim.options, start; info)
    p, _ = opt_multistart(optimize, starts; parallel=optim.parallel, info)
    noise_vars, θ = split(p)
    return θ, nothing, noise_vars
end
function opt_params(loglike::Base.Callable, optim::OptimMLE, model::Nonparametric, noise_var_prior; x_dim::Int, y_dim::Int, info::Bool)
    # TODO: use softplus to ensure positive length scales ?
    split(p) = p[1:y_dim], reshape(p[y_dim+1:end], x_dim, y_dim)
    function ll(p)
        noise_vars, length_scales = split(p)
        loglike(length_scales, noise_vars)
    end

    starts = reduce(hcat, [vcat(
        rand.(noise_priors),
        reduce(vcat, rand.(model.length_scale_priors)),
    ) for _ in 1:optim.multistart])
    bounds = (
        vcat(minimum(noise_var_prior), reduce(vcat, minimum.(model.length_scale_priors))),
        vcat(maximum(noise_var_prior), reduce(vcat, maximum.(model.length_scale_priors)))
    )

    optimize(start) = optim_maximize(ll, bounds, optim.algorithm, optim.options, start; info)
    p, _ = opt_multistart(optimize, starts; parallel=optim.parallel, info)
    noise_vars, length_scales = split(p)
    return nothing, length_scales, noise_vars
end
function opt_params(loglike::Base.Callable, optim::OptimMLE, model::Semiparametric, noise_var_prior; x_dim::Int, y_dim::Int, info::Bool)
    # TODO: use softplus to ensure positive length scales ?
    θ_len = param_count(model.parametric)
    split(p) = p[1:y_dim], p[y_dim+1:y_dim+θ_len], reshape([y_dim+θ_len+1:end], x_dim, y_dim)
    function ll(p)
        noise_vars, θ, length_scales = split(p)
        loglike(θ, length_scales, noise_vars)
    end

    starts = reduce(hcat, [vcat(
        rand.(noise_priors),
        rand.(model.param_priors),
        reduce(vcat, rand.(model.length_scale_priors)),
    ) for _ in 1:optim.multistart])
    bounds = (
        vcat(minimum(noise_var_prior), minimum.(model.param_priors), reduce(vcat, minimum.(model.length_scale_priors))),
        vcat(maximum(noise_var_prior), maximum.(model.param_priors), reduce(vcat, maximum.(model.length_scale_priors)))
    )

    optimize(start) = optim_maximize(ll, bounds, optim.algorithm, optim.options, start; info)
    p, _ = opt_multistart(optimize, starts; parallel=optim.parallel, info)
    noise_vars, θ, length_scales = split(p)
    return θ, length_scales, noise_vars
end


# - - - - - - - - Acquisition Maximization - - - - - - - -

struct OptimMaximizer{
    A<:Optim.AbstractOptimizer,
    O<:Optim.Options,
} <: AcquisitionMaximizer
    algorithm::A
    options::O
    multistart::Int
    parallel::Bool
end

function maximize_acquisition(optim::OptimMaximizer, problem::OptimizationProblem, acq::Base.Callable; info::Bool)
    starts = generate_starts_LHC(get_bounds(cons), multistart)
    optimize(start) = optim_maximize(acq, problem.cons, optim.algorithm, optim.options, start; info)
    arg, _ = opt_multistart(optimize, starts; parallel=optim.parallel, info)
    return arg
end


# - - - - - - - - Utils - - - - - - - -

function optim_maximize(
    f::Base.Callable,
    cons,
    alg::Optim.AbstractOptimizer,
    options::Optim.Options,
    start::AbstractArray{<:Real};
    info::Bool,
)
    result = Optim.optimize(p -> -f(p), cons, start, alg, options)
    info && check_convergence(result)
    return Optim.minimizer(result), -Optim.minimum(result)
end

function check_convergence(result::Optim.OptimizationResults)
    Optim.x_converged(result) || @warn "Optimization run did not converge!"
end

function get_bounds(domain::Optim.TwiceDifferentiableConstraints)
    domain_lb = domain.bounds.bx[1:2:end]
    domain_ub = domain.bounds.bx[2:2:end]
    return domain_lb, domain_ub
end









# UNUSED

# IPNewton cannot handle `start == bound`.
# (https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/ipnewton_basics/#generic-nonlinear-constraints)
function IPNewton_check_start_!(start, bounds, alpha; info=true)
    lb, ub = bounds
    @assert all(ub .- lb .>= 2*alpha)
    @assert all(start .>= lb) && all(start .<= ub)

    lb_far = ((start .- lb) .>= alpha) 
    ub_far = ((ub .- start) .>= alpha)
    all(lb_far) && all(ub_far) && return start
    
    info && @warn "Start is too close to the domain bounds. Moving it further."
    for i in eachindex(start)
        lb_far[i] || (start[i] = lb[i] + alpha)
        ub_far[i] || (start[i] = ub[i] - alpha)
    end
    return start
end
