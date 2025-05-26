
@testset "make_discrete(domain)" begin
    @param_test BOSS.make_discrete begin   
        @params Domain(;
            bounds = ([0., 0.], [1., 1.]),
            discrete = [false, false],
            cons = nothing,
        )
        @success (
            out == in[1]
        )

        @params Domain(;
            bounds = ([0., 0.], [1., 1.]),
            discrete = [true, true],
            cons = nothing,
        )
        @success (
            out == in[1]
        )
        
        @params Domain(;
            bounds = ([0., 0.], [1., 1.]),
            discrete = [false, false],
            cons = x -> x,
        )
        @success (
            out == in[1],
            out.cons([1.2, 1.2]) == [1.2, 1.2],
        )

        @params Domain(;
            bounds = ([0., 0.], [1., 1.]),
            discrete = [false, true],
            cons = x -> x,
        )
        @success (
            out isa Domain,
            out.cons([1.2, 1.2]) == [1.2, 1.],
        )
    end
end

@testset "in_domain(x, domain)" begin
    bounds = Domain(;
        bounds = ([0., 0.], [10., 10.]),
    )
    discrete = Domain(;
        bounds = ([0., 0.], [10., 10.]),
        discrete = [true, false],
    )
    cons = Domain(;
        bounds = ([0., 0.], [10., 10.]),
        cons = x -> [5. - x[2]],  # x[2] <= 5.
    )
    complex = Domain(
        bounds = ([0., 0.], [10., 10.]),
        discrete = [false, true],
        cons = x -> [5. - x[2]],  # x[2] <= 5.
    )


    @param_test BOSS.in_domain begin
        @params [0., 0.], bounds
        @params [5., 5.], bounds
        @params [10., 10.], bounds
        @params [0., 10.], bounds
        @success out == true

        @params [5., -1.], bounds
        @params [5., 11.], bounds
        @success out == false

        @params [0., 0.], discrete
        @params [5., 5.], discrete
        @params [5., 5.2], discrete
        @params [10., 10.], discrete
        @params [0., 10.], discrete
        @success out == true

        @params [5., -1.], discrete
        @params [5., 11.], discrete
        @params [-1., 5.], discrete
        @params [11., 5.], discrete
        @params [5.2, 5.2], discrete
        @params [-0.1, 5.], discrete
        @params [10.1, 5.], discrete
        @success out == false

        @params [0., 0.], cons
        @params [5., 3.], cons
        @params [5., 5.], cons
        @params [10., 5.], cons
        @params [10., 0.], cons
        @success out == true

        @params [5., -1.], cons
        @params [5., 11.], cons
        @params [5., 6.], cons
        @params [-1., 5.], cons
        @params [11., 5.], cons
        @params [11., 6.], cons
        @success out == false

        @params [0., 0.], complex
        @params [5., 3.], complex
        @params [5., 5.], complex
        @params [5.2, 5.], complex
        @params [10., 5.], complex
        @params [10., 0.], complex
        @success out == true

        @params [5., -1.], complex
        @params [5., 11.], complex
        @params [5., 6.], complex
        @params [-1., 5.], complex
        @params [11., 5.], complex
        @params [11., 6.], complex
        @params [5., 4.9], complex
        @params [-1., 4.9], complex
        @success out == false
    end
end

@testset "in_bounds(x, bounds)" begin
    @param_test BOSS.in_bounds begin
        @params [0.], ([0.], [10.])
        @params [5.], ([0.], [10.])
        @params [10.], ([0.], [10.])
        @params [0., 5.], ([0., 0.], [10., 10.])
        @params [5., 5.], ([0., 0.], [10., 10.])
        @params [10., 5.], ([0., 0.], [10., 10.])
        @params [0., 10.], ([0., 0.], [10., 10.])
        @success out == true

        @params [-1.], ([0.], [10.])
        @params [10.1], ([0.], [10.])
        @params [-1., 5.], ([0., 0.], [10., 10.])
        @params [10.1, 5.], ([0., 0.], [10., 10.])
        @params [-1., 11.], ([0., 0.], [10., 10.])
        @success out == false
    end
end

@testset "in_discrete(x, discrete)" begin
    @param_test BOSS.in_discrete begin
        @params [1.], [true]
        @params [1.], [false]
        @params [1.2], [false]
        @params [1., 1.], [true, true]
        @params [1., 1.], [false, true]
        @params [1.2, 1.], [false, true]
        @success out == true

        @params [1.2], [true]
        @params [1., 1.2], [false, true]
        @params [1.2, 1.2], [false, true]
        @success out == false
    end
end

@testset "in_cons(x, cons)" begin
    @param_test BOSS.in_cons begin
        @params [1.], nothing
        @params [1., 1.], nothing
        @success out == true

        @params [1.], x -> x
        @params [0.], x -> x
        @params [1., 1.], x -> [x[1] + x[2]]
        @params [1., -1.], x -> [x[1] + x[2]]
        @success out == true

        @params [-1.], x -> x
        @params [0., -1.], x -> [x[1] + x[2]]
        @success out == false
    end
end
