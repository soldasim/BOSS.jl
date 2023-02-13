using Optim


# - - - - - - - - Model Parameter MLE - - - - - - - -

# TODO: doc
# TODO: into docs: `Optim.Fminbox` -> bounds(priors), not `Optim.Fminbox` -> unconstrained
struct OptimMLE{
    A<:Optim.AbstractOptimizer,
    O<:Optim.Options,
} <: ModelFitter{MLE}
    algorithm::A
    options::O
    multistart::Int
    parallel::Bool

    function OptimMLE(a::A, o::O, m::Int, p::Bool) where {A<:Optim.AbstractOptimizer, O<:Optim.Options}
        (A <: Optim.Fminbox) || @warn "Unconstrained optimization algorithm provided.\nAssuming θ ∈ R for all model parameters."
        new{A,O}(a,o,m,p)
    end
end
OptimMLE(;
    algorithm=Fminbox(LBFGS()),
    options=Optim.Options(),
    multistart=200,
    parallel=true,
) = OptimMLE(algorithm, options, multistart, parallel)

function estimate_parameters(opt::OptimMLE, problem::OptimizationProblem; info::Bool)
    loglike = model_loglike(problem.model, problem.noise_var_priors, problem.data)
    θ, length_scales, noise_vars = opt_params(loglike, opt, problem.model, problem.noise_var_priors; x_dim=x_dim(problem), y_dim=y_dim(problem), info)
    return (θ=θ, length_scales=length_scales, noise_vars=noise_vars)
end

# With `Optim.Fminbox` algorithm
function opt_params(
    loglike::Base.Callable,
    optim::OptimMLE{A},
    model::Parametric,
    noise_var_priors::AbstractArray;
    x_dim::Int,
    y_dim::Int,
    info::Bool
) where {A<:Optim.Fminbox}
    split(p) = p[1:y_dim], p[y_dim+1:end]
    function ll(p)
        noise_vars, θ = split(p)
        loglike(θ, noise_vars)
    end

    starts = reduce(hcat, [vcat(
        rand.(noise_var_priors),
        rand.(model.param_priors),
    ) for _ in 1:optim.multistart])
    bounds = (
        vcat(minimum.(noise_var_priors), minimum.(model.param_priors)),
        vcat(maximum.(noise_var_priors), maximum.(model.param_priors))
    )

    optimize(start) = optim_maximize(ll, bounds, optim.algorithm, optim.options, start)
    p, _ = opt_multistart(optimize, starts, optim.parallel, info)
    noise_vars, θ = split(p)
    return θ, nothing, noise_vars
end
function opt_params(
    loglike::Base.Callable,
    optim::OptimMLE{A},
    model::Nonparametric,
    noise_var_priors::AbstractArray;
    x_dim::Int,
    y_dim::Int,
    info::Bool
) where {A<:Optim.Fminbox}
    split(p) = p[1:y_dim], reshape(p[y_dim+1:end], x_dim, y_dim)
    function ll(p)
        noise_vars, length_scales = split(p)
        loglike(length_scales, noise_vars)
    end

    starts = reduce(hcat, [vcat(
        rand.(noise_var_priors),
        rand.(model.length_scale_priors) |> x->reduce(vcat,x),
    ) for _ in 1:optim.multistart])
    bounds = (
        vcat(minimum.(noise_var_priors), minimum.(model.length_scale_priors) |> x->reduce(vcat,x)),
        vcat(maximum.(noise_var_priors), maximum.(model.length_scale_priors) |> x->reduce(vcat,x))
    )

    optimize(start) = optim_maximize(ll, bounds, optim.algorithm, optim.options, start)
    p, _ = opt_multistart(optimize, starts, optim.parallel, info)
    noise_vars, length_scales = split(p)
    return nothing, length_scales, noise_vars
end
function opt_params(
    loglike::Base.Callable,
    optim::OptimMLE{A},
    model::Semiparametric,
    noise_var_priors::AbstractArray;
    x_dim::Int,
    y_dim::Int,
    info::Bool
) where {A<:Optim.Fminbox}
    θ_len = param_count(model.parametric)
    split(p) = p[1:y_dim], p[y_dim+1:y_dim+θ_len], reshape(p[y_dim+θ_len+1:end], x_dim, y_dim)
    function ll(p)
        noise_vars, θ, length_scales = split(p)
        loglike(θ, length_scales, noise_vars)
    end

    starts = reduce(hcat, [vcat(
        rand.(noise_var_priors),
        rand.(model.parametric.param_priors),
        rand.(model.nonparametric.length_scale_priors) |> x->reduce(vcat,x),
    ) for _ in 1:optim.multistart])
    bounds = (
        vcat(minimum.(noise_var_priors), minimum.(model.parametric.param_priors), minimum.(model.nonparametric.length_scale_priors) |> x->reduce(vcat,x)),
        vcat(maximum.(noise_var_priors), maximum.(model.parametric.param_priors), maximum.(model.nonparametric.length_scale_priors) |> x->reduce(vcat,x))
    )

    optimize(start) = optim_maximize(ll, bounds, optim.algorithm, optim.options, start)
    p, _ = opt_multistart(optimize, starts, optim.parallel, info)
    noise_vars, θ, length_scales = split(p)
    return θ, length_scales, noise_vars
end

# With unconstrained algorithm
function opt_params(
    loglike::Base.Callable,
    optim::OptimMLE,
    model::Parametric,
    noise_var_priors::AbstractArray;
    x_dim::Int,
    y_dim::Int,
    info::Bool
)
    # `softplus` used to keep noise variance positive
    split(p) = softplus.(p[1:y_dim]), p[y_dim+1:end]
    function ll(p)
        noise_vars, θ = split(p)
        loglike(θ, noise_vars)
    end

    starts = reduce(hcat, [vcat(
        rand.(noise_var_priors) .|> invsoftplus,
        rand.(model.param_priors),
    ) for _ in 1:optim.multistart])

    optimize(start) = optim_maximize(ll, nothing, optim.algorithm, optim.options, start)
    p, _ = opt_multistart(optimize, starts, optim.parallel, info)
    noise_vars, θ = split(p)
    return θ, nothing, noise_vars
end
function opt_params(
    loglike::Base.Callable,
    optim::OptimMLE,
    model::Nonparametric,
    noise_var_priors::AbstractArray;
    x_dim::Int,
    y_dim::Int,
    info::Bool
)
    # `softplus` used to keep noise variance and length scales positive
    split(p) = softplus.(p[1:y_dim]), softplus.(reshape(p[y_dim+1:end], x_dim, y_dim))
    function ll(p)
        noise_vars, length_scales = split(p)
        loglike(length_scales, noise_vars)
    end

    starts = reduce(hcat, [vcat(
        rand.(noise_var_priors) .|> invsoftplus,
        rand.(model.length_scale_priors) |> x->reduce(vcat,x) .|> invsoftplus,
    ) for _ in 1:optim.multistart])

    optimize(start) = optim_maximize(ll, nothing, optim.algorithm, optim.options, start)
    p, _ = opt_multistart(optimize, starts, optim.parallel, info)
    noise_vars, length_scales = split(p)
    return nothing, length_scales, noise_vars
end
function opt_params(
    loglike::Base.Callable,
    optim::OptimMLE,
    model::Semiparametric,
    noise_var_priors::AbstractArray;
    x_dim::Int,
    y_dim::Int,
    info::Bool
)
    θ_len = param_count(model.parametric)
    # `softplus` used to keep noise variance and length scales positive
    split(p) = softplus.(p[1:y_dim]), p[y_dim+1:y_dim+θ_len], softplus.(reshape(p[y_dim+θ_len+1:end], x_dim, y_dim))
    function ll(p)
        noise_vars, θ, length_scales = split(p)
        loglike(θ, length_scales, noise_vars)
    end

    starts = reduce(hcat, [vcat(
        rand.(noise_var_priors) .|> invsoftplus,
        rand.(model.parametric.param_priors),
        rand.(model.nonparametric.length_scale_priors) |> x->reduce(vcat,x) .|> invsoftplus,
    ) for _ in 1:optim.multistart])

    optimize(start) = optim_maximize(ll, nothing, optim.algorithm, optim.options, start)
    p, _ = opt_multistart(optimize, starts, optim.parallel, info)
    noise_vars, θ, length_scales = split(p)
    return θ, length_scales, noise_vars
end


# - - - - - - - - Acquisition Maximization - - - - - - - -

# TODO: doc
struct OptimMaximizer{
    A<:Optim.AbstractOptimizer,
    O<:Optim.Options,
} <: AcquisitionMaximizer
    algorithm::A
    options::O
    multistart::Int
    parallel::Bool
end
OptimMaximizer(;
    algorithm=IPNewton(),
    options=Optim.Options(),
    multistart=200,
    parallel=true,
) = OptimMaximizer(algorithm, options, multistart, parallel)

struct OptimDomain{
    C<:Union{Optim.AbstractConstraints, Tuple, Nothing}
} <: Domain
    cons::C
end
get_bounds(domain::OptimDomain) = get_bounds(domain.cons)
in_domain(domain::OptimDomain, x) = in_domain(domain.cons, x)

function maximize_acquisition(optim::OptimMaximizer, problem::OptimizationProblem, acq::Base.Callable; info::Bool)
    starts = generate_starts_LHC(get_bounds(problem.domain), optim.multistart)
    optimize(start) = optim_maximize(acq, problem.domain.cons, optim.algorithm, optim.options, start)
    arg, _ = opt_multistart(optimize, starts, optim.parallel, info)
    return arg
end


# - - - - - - - - Utils - - - - - - - -

function optim_maximize(
    f::Base.Callable,
    cons::Optim.AbstractConstraints,
    alg::Optim.AbstractOptimizer,
    options::Optim.Options,
    start::AbstractArray{<:Real},
)
    result = Optim.optimize(p -> -f(p), cons, start, alg, options)
    return Optim.minimizer(result), -Optim.minimum(result)
end
function optim_maximize(
    f::Base.Callable,
    bounds::Tuple,
    alg::Optim.Fminbox,
    options::Optim.Options,
    start::AbstractArray{<:Real},
)
    result = Optim.optimize(p -> -f(p), bounds..., start, alg, options)
    return Optim.minimizer(result), -Optim.minimum(result)
end
function optim_maximize(
    f::Base.Callable,
    ::Nothing,
    alg::Optim.AbstractOptimizer,
    options::Optim.Options,
    start::AbstractArray{<:Real},
)
    result = Optim.optimize(p -> -f(p), start, alg, options)
    return Optim.minimizer(result), -Optim.minimum(result)
end

function get_bounds(domain::Optim.TwiceDifferentiableConstraints)
    domain_lb = domain.bounds.bx[1:2:end]
    domain_ub = domain.bounds.bx[2:2:end]
    return domain_lb, domain_ub
end
get_bounds(domain::Tuple) = domain

in_domain(domain::Optim.TwiceDifferentiableConstraints, x::AbstractArray{<:Real}) =
    Optim.isinterior(domain, x)
function in_domain(domain::Tuple, x::AbstractVector)
    lb, ub = domain
    any(x .< lb) && return false
    any(x .> ub) && return false
    return true
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
