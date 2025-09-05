
function safe_model_loglike(model::SurrogateModel, data::ExperimentData; options::BossOptions=BossOptions())
    ll = model_loglike(model, data)
    ll_safe = make_safe(ll, -Inf; options.info, options.debug)
    return ll_safe
end

function safe_data_loglike(model::SurrogateModel, data::ExperimentData; options::BossOptions=BossOptions())
    ll = data_loglike(model, data)
    ll_safe = make_safe(ll, -Inf; options.info, options.debug)
    return ll_safe
end
