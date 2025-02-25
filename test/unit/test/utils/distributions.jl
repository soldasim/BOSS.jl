
@testset "calc_inverse_gamma(lb, ub)" begin
    @param_test calc_inverse_gamma begin
        @params 0., 1.
        @params 5., 8.
        @params 0., 1e-8
        @params 0., 1e8
        @success BOSS.cdf(out, in[2]) - BOSS.cdf(out, in[1]) > 0.98

        @params 0., 0.
        @params 8., 5.
        @params -1., 1.
        @params 1., -1.
        @params -2., -1.
        @failure
    end
end

@testset "mvnormal(μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real})" begin
    @param_test BOSS.mvnormal begin
        @params [0.], [1.]
        @params zeros(10), ones(10)
        @params [1,2,3], [1,2,3]
        @success (
            out isa MvNormal,
            out.μ == in[1],
            out.Σ == Diagonal(in[2] .^ 2),
        )
    end
end

@testset "mvlognormal(μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real})" begin
    @param_test BOSS.mvlognormal begin
        @params [0.], [1.]
        @params zeros(10), ones(10)
        @params [1,2,3], [1,2,3]
        @success (
            out isa MvLogNormal,
            out.normal.μ == in[1],
            out.normal.Σ == Diagonal(in[2] .^ 2),
        )
    end
end
