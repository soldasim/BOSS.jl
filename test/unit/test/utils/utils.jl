
@testset "ith(i)" begin
    v = [1,2,3]
    vs = [[11,12,13],[21,22,23],[31,32,33]]

    @test BOSS.ith(2)(v) == 2
    @test BOSS.ith(2).(vs) == [12,22,32]
end

@testset "cond_func(f)" begin
    f(x) = 0.

    @test BOSS.cond_func(f)(5., false) == 5.
    @test BOSS.cond_func(f)(5., true) == 0.
    @test BOSS.cond_func(f).([5., 5.], [false, true]) == [5., 0.]
end

@testset "discrete_round(dims, x)" begin
    @param_test BOSS.discrete_round begin
        @params nothing, [4.2, 5.3]
        @params [false, false], [4.2, 5.3]
        @success out == [4.2, 5.3]

        @params missing, [4.2, 5.3]
        @params [true, true], [4.2, 5.3]
        @success out == [4., 5.]

        @params [true, false], [4.2, 5.3]
        @success out == [4., 5.3]
    end
end

@testset "is_feasible(y, y_max)" begin
    @param_test BOSS.is_feasible begin
        @params [1.], [Inf]
        @params [Inf], [Inf]
        @params [1.], [10.]
        @params [10.], [10.]
        @params [1., 11.], [10., Inf]
        @params [10., 11.], [10., Inf]
        @params [-Inf, 1.], [-Inf, 10.]
        @success out == true

        @params [11.], [10.]
        @params [1., 11.], [10., 10.]
        @params [11., 11.], [10., 10.]
        @params [Inf], [10.]
        @params [Inf], [-Inf]
        @params [0., 1.], [-Inf, 10.]
        @success out == false
    end
end

@testset "exclude_exterior_points(domain, X, Y)" begin
    domain = BOSS.Domain(;
        bounds = ([0., 0.], [10., 10.]),
        discrete = [true, false],
        cons = x -> [x[2] - 5.],  # x[2] >= 5.
    )
    options = BOSS.BossOptions(; info=false)

    @param_test (in...) -> BOSS.exclude_exterior_points(in...; options) begin
        @params deepcopy(domain), [5.;5.;; 5.;5.;; 8.;8.;;], [5.;5.;; 5.;5.;; 8.;8.;;]
        @success out == (in[2], in[3])

        @params deepcopy(domain), [2.;2.;; 5.;5.;; 8.;8.;;], [2.;2.;; 5.;5.;; 8.;8.;;]
        @params deepcopy(domain), [5.2;5.2;; 5.;5.;; 8.;8.;;], [5.2;5.2;; 5.;5.;; 8.;8.;;]
        @params deepcopy(domain), [11.;11.;; 5.;5.;; 8.;8.;;], [11.;11.;; 5.;5.;; 8.;8.;;]
        @success out == ([5.;5.;; 8.;8.;;], [5.;5.;; 8.;8.;;])
    end
end
