
@testset "make_discrete(domain)" begin
    @param_test BOSS.make_discrete begin        
        @params BOSS.Domain(;
            bounds = ([0., 0.], [1., 1.]),
            discrete = [false, false],
            cons = nothing,
        )
        @success (
            out == in[1]
        )

        @params BOSS.Domain(;
            bounds = ([0., 0.], [1., 1.]),
            discrete = [true, true],
            cons = nothing,
        )
        @success (
            out == in[1]
        )
        
        @params BOSS.Domain(;
            bounds = ([0., 0.], [1., 1.]),
            discrete = [false, false],
            cons = x -> x,
        )
        @success (
            out == in[1],
            out.cons([1.2, 1.2]) == [1.2, 1.2],
        )

        @params BOSS.Domain(;
            bounds = ([0., 0.], [1., 1.]),
            discrete = [false, true],
            cons = x -> x,
        )
        @success (
            out isa BOSS.Domain,
            out.cons([1.2, 1.2]) == [1.2, 1.],
        )
    end
end
