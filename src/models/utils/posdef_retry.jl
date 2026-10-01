
"""
The maximum number of retries performed when a covariance matrix is not positive
definite (e.g. due to near-duplicate points in the data). Used by surrogate models
that construct their own GP-like covariance matrices. Each retry adds a larger jitter
(relative to the fitted amplitude) before reconditioning.
"""
const GP_POSTERIOR_MAX_RETRIES = 4

"""
Calls `f(jitter)` with an escalating `jitter` (scaled to `amplitude^2`, up to
[`GP_POSTERIOR_MAX_RETRIES`](@ref) retries), retrying if it throws `PosDefException`
(e.g. because (near-)duplicate points make the covariance matrix singular). `f` is
responsible for building its own covariance matrix/kernel using `jitter` (e.g. added to
the noise diagonal) and performing the operation that may throw. Rethrows after
exhausting all retries.
"""
function _posdef_retry(f, amplitude::Real; context::AbstractString="")
    for attempt in 0:GP_POSTERIOR_MAX_RETRIES
        jitter = iszero(attempt) ? 0.0 : amplitude^2 * 10.0^(attempt - GP_POSTERIOR_MAX_RETRIES - 1)

        try
            return f(jitter)

        catch e
            e isa PosDefException || rethrow(e)
            msg = "Covariance matrix not positive definite ($context, attempt $(attempt+1)/$(GP_POSTERIOR_MAX_RETRIES+1), jitter $jitter)"
            if attempt < GP_POSTERIOR_MAX_RETRIES
                msg *= "; retrying"
                @warn msg
            else
                msg *= "; giving up"
                @warn msg
                rethrow(e)
            end
        end
    end
end
