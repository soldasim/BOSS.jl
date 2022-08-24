using Optim

function construct_acq(ei, feasibility_model; feasibility, best_yet)
    if feasibility
        if isnothing(best_yet)
            return x -> constraint_weighted_acq(1., x, feasibility_model)
        else
            return x -> constraint_weighted_acq(ei(x), x, feasibility_model)
        end
    else
        return ei
    end
end

function constraint_weighted_acq(acq, x, feasibility_model)
    return prod(feasibility_probabilities(feasibility_model)(x)) * acq
end

# 'domain' is either 'TwiceDifferentiableConstraints' or a Tuple of lb and ub
function opt_acq(acq, domain; multistart=1, info=true, debug=false)
    results = Vector{Tuple{Vector{Float64}, Float64}}(undef, multistart)
    convergence_errors = 0
    for i in 1:multistart  # @floop TODO
        try
            opt_res = optim_(x -> -acq(x), domain)
            res = Optim.minimizer(opt_res), -Optim.minimum(opt_res)
            # in_domain(res[1], domain_lb, domain_ub) || throw(ErrorException("Optimization result out of the domain."))
            results[i] = res
        catch e
            debug && throw(e)
            @reduce convergence_errors += 1
            results[i] = ([], -Inf)
        end
    end

    info && (convergence_errors > 0) && print("      $(convergence_errors)/$(multistart) optimization runs failed to converge!\n")
    opt_i = argmax([res[2] for res in results])
    return results[opt_i]
end

function optim_(acq, domain)
    domain_lb, domain_ub = domain
    start = generate_starting_point_(domain_lb, domain_ub)
    return Optim.optimize(acq, domain[1], domain[2], start, Fminbox(LBFGS()))
end
function optim_(acq, constraints::TwiceDifferentiableConstraints)
    domain_lb, domain_ub = get_bounds(constraints)

    # TODO better starting point generation ?
    start = nothing
    while isnothing(start) || (!Optim.isinterior(constraints, start))
        start = generate_starting_point_(domain_lb, domain_ub)
    end

    return Optim.optimize(acq, constraints, start, IPNewton())
end

function get_bounds(constraints::TwiceDifferentiableConstraints)
    domain_lb = constraints.bounds.bx[1:2:end]
    domain_ub = constraints.bounds.bx[2:2:end]
    return domain_lb, domain_ub
end

function generate_starting_point_(domain_lb, domain_ub)
    dim = length(domain_lb)
    start = rand(dim) .* (domain_ub .- domain_lb) .+ domain_lb
    return start
end

function in_domain(x, domain_lb, domain_ub)
    any(x .< domain_lb) && return false
    any(x .> domain_ub) && return false
    return true
end

function EI(x, fitness::LinFitness, model, ϵ_samples; best_yet, sample_count)
    μy, Σy = model[1](x), model[2](x)

    μf = fitness.coefs' * μy
    σf = sqrt((fitness.coefs .^ 2)' * (Σy .^ 2))
    
    norm_ϵ = (μf - best_yet) / σf
    return (μf - best_yet) * cdf(Distributions.Normal(), norm_ϵ) + σf * pdf(Distributions.Normal(), norm_ϵ)
end

# TODO try run with NonlinFitness
function EI(x, fitness::NonlinFitness, model, ϵ_samples; best_yet, sample_count)
    μy, Σy = model[1](x), model[2](x)
    pred_samples = [μy .+ (Σy .* ϵ_samples[i,:]) for i in 1:sample_count]
    return sum(max.(0, fitness.(pred_samples) .- best_yet)) / sample_count
end
