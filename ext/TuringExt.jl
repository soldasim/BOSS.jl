module TuringExt

using BOSS
using Turing
using Distributions

"""
Implementation of the abstract `BOSS.TuringBI`. See the docs `? BOSS.TuringBI`.
"""
@kwdef struct TuringBI{S} <: BOSS.TuringBI
    sampler::S = PG(20)
    warmup::Int = 400
    samples_in_chain::Int = 20
    chain_count::Int = 4
    leap_size::Int = 5
    parallel::Bool = false
end

BOSS.TuringBI(args...; kwargs...) = TuringBI(args...; kwargs...)

total_samples(t::TuringBI) = t.chain_count * t.samples_in_chain

function BOSS.estimate_parameters(turing::TuringBI, problem::BossProblem, options::BossOptions; return_all::Bool=false)
    if BOSS.sliceable(problem.model)
        # In case of sliceable model, handle each y dimension separately.
        y_dim_ = BOSS.y_dim(problem)
        problem_slices = BOSS.slice.(Ref(problem), 1:y_dim_)
        
        results = _estimate_parameters.(Ref(turing), problem_slices, Ref(options); return_all)
        return reduce_slice_results(results)
    
    else
        # In case of non-sliceable model, optimize all parameters simultaneously.
        return _estimate_parameters(turing, problem, options; return_all)
    end
end

function _estimate_parameters(turing::TuringBI, problem::BossProblem, options::BossOptions; return_all::Bool=false)
    params = BOSS.params_sampler(problem.model, problem.data)()
    
    tm = turing_model(problem.model, params, problem.data; options)
    
    # if all parameters have Dirac priors
    if isnothing(tm)
        return BOSS.BIParams([deepcopy(params) for _ in 1:total_samples(turing)])
    end
    
    chains = sample_chains(turing, tm)
    samples = devec_chains(chains, problem.model, params, problem.data)

    return BOSS.BIParams(samples)
end

# function _estimate_parameters(turing::TuringBI, problem::BossProblem, options::BossOptions; return_all::Bool=false)
#     @info "--- REJECTION SAMPLING ---"
#     # TODO
#     # sample from prior
#     sampler = BOSS.params_sampler(problem.model, problem.data)
#     samples = [sampler() for _ in 1:10_000]

#     # resample according to the likelihood
#     data_loglike = BOSS.safe_data_loglike(problem.model, problem.data; options)
#     lls = data_loglike.(samples)
#     likes = exp.(lls)
#     samples_ = sample(samples, Distributions.StatsBase.ProbabilityWeights(likes), 120)

#     return BOSS.BIParams(samples_)
# end

function turing_model(model::SurrogateModel, params::ModelParams, data::ExperimentData; options::BossOptions)
    vec_, devec_ = BOSS.vectorizer(model, data)
    params_prior_ = BOSS.ModelParamsPrior(model, data)

    if length(params_prior_) == 0
        return nothing
    end

    data_loglike = BOSS.safe_data_loglike(model, data; options)
    data_loglike_(ps) = data_loglike(devec_(params, ps))

    return turing_model(params_prior_, data_loglike_)
end
Turing.@model function turing_model(params_prior::BOSS.ModelParamsPrior, data_loglike::Function)
    ps ~ params_prior
    Turing.@addlogprob! data_loglike(ps)
end

function sample_chains(turing::TuringBI, turing_model)
    samples_in_chain = turing.warmup + (turing.leap_size * turing.samples_in_chain)
    if turing.parallel
        chains = Turing.sample(turing_model, turing.sampler, MCMCThreads(), samples_in_chain, turing.chain_count; progress=false)
    else
        chains = mapreduce(_ -> Turing.sample(turing_model, turing.sampler, samples_in_chain; progress=false), Turing.AbstractMCMC.chainscat, 1:turing.chain_count)
    end
    # VNChain (Turing ≥0.45) stores each parameter as a Matrix{T}(n_iters, n_chains).
    # `ps ~ ModelParamsPrior` is always multivariate, so entries are Vector{Float64}.
    ps_key = only(k for k in keys(chains._data) if k isa Turing.FlexiChains.Parameter)
    ps_matrix = chains._data[ps_key]  # (n_iters, n_chains)
    n = size(ps_matrix, 1)
    idx = (turing.warmup + turing.leap_size):turing.leap_size:n
    return [ps_matrix[i, j] for j in axes(ps_matrix, 2) for i in idx]
end

function devec_chains(ps_samples::AbstractVector, model::SurrogateModel, params::ModelParams, data::ExperimentData)
    vec_, devec_ = BOSS.vectorizer(model, data)
    return devec_.(Ref(params), ps_samples)
end

function reduce_slice_results(results::AbstractVector{<:BOSS.BIParams})
    @assert allequal(length.(results))
    sample_count = length(first(results))

    samples = [BOSS.join_slices(getindex.(results, Ref(i))) for i in 1:sample_count]
    return BOSS.BIParams(samples)
end

# workaround for Turing erroring with `Bijector`s without defined `with_logabsdet_jacobian` method
function Turing.with_logabsdet_jacobian(b::Turing.Bijector, x)
    y = Turing.transform(b, x)
    ladj = Turing.logabsdetjac(b, x)
    return y, ladj
end

end # module TuringExt
