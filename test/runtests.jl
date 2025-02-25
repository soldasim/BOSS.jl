using BOSS
using Test
using Aqua

using Turing
using LinearAlgebra

```
Determines whether parallelization of BOSS is allowed during tests.
```
# Currently, enabling parallel testing causes `StackOverflowError`s on Ubuntu.
# See https://github.com/libprima/PRIMA.jl/issues/25
const PARALLEL_TESTS = false

@testset "BOSS TESTS" verbose=true begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(BOSS)
    end

    include("unit/runtests.jl")

    include("combinatorial/runtests.jl")
end
