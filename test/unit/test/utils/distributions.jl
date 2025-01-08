
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
