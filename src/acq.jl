using Optim

function construct_acq(ei, c_model; constrained, best_yet)
    if constrained
        if isnothing(best_yet)
            return x -> constraint_weighted_acq(1., x, c_model)
        else
            return x -> constraint_weighted_acq(ei(x), x, c_model)
        end
    else
        return ei
    end
end

function constraint_weighted_acq(acq, x, c_model)
    return prod(constraint_probabilities(c_model)(x)) * acq
end

function opt_acq(acq, domain_lb, domain_ub; multistart=1, info=true, debug=false)
    dim = length(domain_lb)
    starts = rand(dim, multistart) .* (domain_ub .- domain_lb) .+ domain_lb

    results = Vector{Tuple{Vector{Float64}, Float64}}(undef, multistart)
    convergence_errors = 0
    @floop for i in 1:multistart
        try
            opt_res = Optim.optimize(x -> -acq(x), domain_lb, domain_ub, starts[:,i], Fminbox(LBFGS()))
            res = Optim.minimizer(opt_res), -Optim.minimum(opt_res)
            in_domain(res[1], domain_lb, domain_ub) || throw(ErrorException("Optimization result out of the domain."))
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

function in_domain(x, domain_lb, domain_ub)
    any(x .< domain_lb) && return false
    any(x .> domain_ub) && return false
    return true
end

# Used when: model posterior predictive distribution is Gaussian
#            and fitness is linear
function EI_gauss(x, fitness::LinFitness, model, noise_samples, ϵ_samples; best_yet, sample_count)
    μy, Σy = model[1](x), model[2](x)

    μf = fitness.coefs' * μy
    σf = sqrt((fitness.coefs .^ 2)' * (Σy .^ 2))
    
    norm_ϵ = (μf - best_yet) / σf
    return (μf - best_yet) * cdf(Distributions.Normal(), norm_ϵ) + σf * pdf(Distributions.Normal(), norm_ϵ)
end

# Used when: model posterior predictive distribution is Gaussian
#            but fitness is NOT linear
function EI_gauss(x, fitness::NonlinFitness, model, noise_samples, ϵ_samples; best_yet, sample_count)
    μy, Σy = model[1](x), model[2](x)
    pred_samples = [μy .+ (noise_samples[i] .* ϵ_samples[i]) for i in 1:sample_count]
    return EI_MC(fitness, pred_samples; sample_count, best_yet)
end

# Used when: model posterior predictive distribution is NOT Gaussian
function EI_nongauss(x, fitness::Fitness, model_predict, param_samples, noise_samples, ϵ_samples; best_yet, sample_count)
    pred_samples = [model_predict(x, param_samples[i]) .+ (noise_samples[i] .* ϵ_samples[i]) for i in 1:sample_count]
    return EI_MC(fitness, pred_samples; sample_count, best_yet)
end

function EI_MC(fitness::Fitness, pred_samples; sample_count, best_yet)
    return sum(max.(0, fitness.(pred_samples) .- best_yet)) / sample_count
end
