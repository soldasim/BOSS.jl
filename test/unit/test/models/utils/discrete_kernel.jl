
@testset "(::DiscreteKernel)(x1, x2)" begin
    @param_test BOSS.DiscreteKernel begin
        @params Matern32Kernel(), [false, false]
        @success (
            out([1.2, 1.2], [3.8, 3.8]) == Matern32Kernel()([1.2, 1.2], [3.8, 3.8]),
            out([1.2, 2.], [3.8, 4.]) == Matern32Kernel()([1.2, 2.], [3.8, 4.]),
        )

        @params Matern32Kernel(), [false, true]
        @success (
            out([1.2, 1.2], [3.8, 3.8]) == out([1.2, 1.], [3.8, 4.]),
            out([1.2, 1.2], [3.8, 4.]) == out([1.2, 1.], [3.8, 4.]),
        )
    end
end

@testset "make_discrete(kernel, discrete)" begin
    @param_test BOSS.make_discrete begin
        @params Matern32Kernel(), [false, false]
        @success (
            out isa BOSS.DiscreteKernel,
            out.kernel == in[1],
            out([1.2, 1.2], [3.8, 3.8]) != out([1.2, 1.], [3.8, 4.]),
            out([1.2, 1.2], [3.8, 3.8]) == in[1]([1.2, 1.2], [3.8, 3.8]),
        )

        @params Matern32Kernel(), [false, true]
        @success (
            out isa BOSS.DiscreteKernel,
            out.kernel == in[1],
            out([1.2, 1.2], [3.8, 3.8]) == out([1.2, 1.], [3.8, 4.]),
            out([1.2, 1.2], [3.8, 3.8]) != in[1]([1.2, 1.2], [3.8, 3.8]),
        )
    end
end
